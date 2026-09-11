use std::f32::NEG_INFINITY;

use wasm_bindgen::prelude::*;
use web_sys::{FileReader, HtmlImageElement, HtmlInputElement, HtmlCanvasElement, Event, CanvasRenderingContext2d};
use wasm_bindgen::JsCast;
use tract_onnx::prelude::*;

static MODEL_BYTES: &[u8] = include_bytes!("model/galaxy_mnist_model.onnx");

fn extract_pixels(img: &HtmlImageElement) -> Result<Vec<f32>, JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas: HtmlCanvasElement = document
        .create_element("canvas")?
        .dyn_into()?;

    let target_size = 224.0;
    canvas.set_width(224);
    canvas.set_height(224);

    let ctx:CanvasRenderingContext2d = canvas
        .get_context("2d")?
        .unwrap()
        .dyn_into()?;


    ctx.set_fill_style_str("black");
    ctx.fill_rect(0.0, 0.0, target_size, target_size);

    let img_w = img.natural_width() as f64;
    let img_h = img.natural_height() as f64;

    let scale = (target_size / img_w).min(target_size/img_h);
    let draw_w = img_w * scale;
    let draw_h = img_h * scale;

    let dx = (target_size - draw_w) / 2.0;
    let dy = (target_size - draw_h) / 2.0;

    ctx.draw_image_with_html_image_element_and_dw_and_dh(
        img, dx, dy, draw_w, draw_h,
    )?;

    let image_data = ctx.get_image_data(0.0, 0.0, 224.0, 224.0)?;
    let raw_bytes = image_data.data();

    let mut chw_pixels = vec![0.0f32; 3 * 224 * 224];
    for i in 0..(224 * 224) {

        let r = (raw_bytes[i * 4] as f32 / 255.0 - 0.5) / 0.5;
        let g = (raw_bytes[i * 4 + 1] as f32 / 255.0 - 0.5) / 0.5;
        let b = (raw_bytes[i * 4 + 2] as f32 / 255.0 - 0.5) / 0.5;


        chw_pixels[i] = r;                  
        chw_pixels[224 * 224 + i] = g;      
        chw_pixels[2 * 224 * 224 + i] = b;  
    }

    Ok(chw_pixels)
}

#[wasm_bindgen(start)]
pub fn run() -> Result<(), JsValue> {
    let window = web_sys::window().expect("no global window exists");
    let document = window.document().expect("should have document on window");

    // load ONNX model
    let model = tract_onnx::onnx()
        .model_for_read(&mut &MODEL_BYTES[..])
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .with_input_fact(
            0, 
            InferenceFact::dt_shape(f32::datum_type(), vec![1, 3, 224, 224]),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .into_optimized()
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .into_runnable()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // get input element
    let input: HtmlInputElement = document
        .get_element_by_id("galaxy-image")
        .expect("element #galaxy-image missing")
        .dyn_into()?;

    // event listener closure
    let closure = Closure::<dyn FnMut(_)>::new(move |event: Event| {
        let target = event.target().unwrap();
        let input_elem: HtmlInputElement = target.dyn_into().unwrap();

        if let Some(files) = input_elem.files() {
            if let Some(file) = files.get(0) {
                web_sys::console::log_1(&format!("Selected image: {}", file.name()).into());

                let reader = FileReader::new().unwrap();
                let reader_clone = reader.clone();
                let model_clone = model.clone();

                // file loaded event
                let onload = Closure::<dyn FnMut(_)>::new(move |_e: Event| {
                    if let Ok(result) = reader_clone.result() {
                        if let Some(data_url) = result.as_string() {
                            let doc = web_sys::window().unwrap().document().unwrap();
                            if let Some(img_elem) = doc.get_element_by_id("image-preview") {
                                let img: HtmlImageElement = img_elem.dyn_into().unwrap();
                                let img_clone = img.clone();
                                let model_inner_clone = model_clone.clone();

                                let img_onload = Closure::<dyn FnMut(_)>::new(move |_e: Event| {
                                    if let Ok(pixels) = extract_pixels(&img_clone) {
                                        if let Ok(tensor) = Tensor::from_shape(&[1, 3, 224, 224], &pixels) {
                                            let class_names = [
                                                "smooth and round", 
                                                "smooth and cigar-shaped", 
                                                "edge-on disk", 
                                                "unbarred spiral"
                                            ];

                                            if let Ok(result) = model_inner_clone.run(tvec!(tensor.into())) {
                                                if let Ok(logits) = result[0].to_array_view::<f32>() {

                                                    let logits_slice = logits.as_slice().unwrap();

                                                    let max_logit = logits_slice
                                                        .iter()
                                                        .cloned()
                                                        .fold(f32::NEG_INFINITY, f32::max);
                                                        
                                                    let exps: Vec<f32> = logits_slice  
                                                        .iter()
                                                        .map(|&x| (x - max_logit).exp())
                                                        .collect();

                                                    let sum_exps: f32 = exps.iter().sum();

                                                    let probabilities: Vec<f32> = exps
                                                        .iter()
                                                        .map(|&x| (x / sum_exps) * 100.0)
                                                        .collect();

                                                    //most likely prediction:
                                                    let mut top_idx = 0;
                                                    let mut max_pct = -1.0;

                                                    for (i, &pct) in probabilities.iter().enumerate() {
                                                        if pct > max_pct {
                                                            max_pct = pct;
                                                            top_idx = i;
                                                        }
                                                    }

                                                    let top_label = class_names.get(top_idx).unwrap_or(&"unknown");

                                                    let mut html = format!(
                                                        "<div class=\"top-prediction\">most likely: <span>{}</span> ({:.1}%)</div>",
                                                        top_label, max_pct
                                                    );

                                                    html.push_str("<div class=\"bars-container\">");
                                                    for (i, &pct) in probabilities.iter().enumerate() {
                                                        let label = class_names.get(i).unwrap_or(&"unknown");
                                                        html.push_str(&format!(
                                                            "<div class=\"bar-row\">\
                                                                <div class=\"bar-label\"><span>{}</span><span>{:.1}%</span></div>\
                                                                <div class=\"bar-track\">\
                                                                    <div class=\"bar-fill\" style=\"width: {:.1}%;\"></div>\
                                                                </div>\
                                                            </div>",
                                                            label, pct, pct
                                                        ));
                                                    }
                                                    html.push_str("</div>");

                                                    let doc = web_sys::window().unwrap().document().unwrap();
                                                    if let Some(result_elem) = doc.get_element_by_id("prediction-result") {
                                                        result_elem.set_inner_html(&html);
                                                    }

                                                }
                                            }
                                        }
                                    }
                                });

                                img.set_onload(Some(img_onload.as_ref().unchecked_ref()));
                                img_onload.forget();

                                img.set_src(&data_url);
                                img.class_list().remove_1("hidden").unwrap();
                            }
                        }
                    }
                });

                reader.set_onload(Some(onload.as_ref().unchecked_ref()));
                onload.forget();

                reader.read_as_data_url(&file).unwrap();
            }
        }
    });

    input.add_event_listener_with_callback("change", closure.as_ref().unchecked_ref())?;
    closure.forget();

    Ok(())
}
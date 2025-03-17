#![feature(f16)]  // 启用 f16 支持
#![feature(associated_type_defaults)]

mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod api;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::sync::Arc;
use crate::model::Llama;



// fn main() {
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     let llama = model::Llama::<f32>::from_safetensors(&model_dir);
//     let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
//     let input = "Once upon a time";
//     let binding = tokenizer.encode(input, true).unwrap();
//     let input_ids = binding.get_ids();
//     print!("\n{}", input);
//     let output_ids = llama.generate(
//         input_ids,
//         500,
//         0.8,
//         30,
//         1.,
//     );
//     println!("{}", tokenizer.decode(&output_ids, true).unwrap());
// }

// fn main() {
//     let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/story");
//     let model = Llama::<f32>::from_safetensors(&model_path);
//     let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).unwrap();
//     let mut cache = model.new_cache();
//     let mut history = Vec::new();

//     loop {
//         println!("User: ");
//         let mut user_input = String::new();
//         std::io::stdin().read_line(&mut user_input).unwrap();
//         let user_input = user_input.trim();

//         if user_input == "exit" {
//             break;
//         }

//         let response = model.chat(user_input, &tokenizer, &mut history, &mut cache);
//         println!("AI: {}", response);
//     }
// }

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model_dir = std::path::Path::new("models/story");
    
    let model = Arc::new(Llama::from_safetensors(model_dir)); 
    
    let tokenizer = Arc::new(
        Tokenizer::from_file(model_dir.join("tokenizer.json"))  
            .expect("Failed to load tokenizer")
    );
    
    api::run_api_server(model, tokenizer, 8080).await
}



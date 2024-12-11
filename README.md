# **LLaMa 2 Implementation**

This repository contains the PyTorch implementation of the LLaMa 2 model.

## **Features Implemented**
1. Rotary Positional Embeddings  
2. Grouped Query Attention  
3. Key-Value (KV) Cache  
4. RMS Normalization  
5. SwiGLU Activation  
6. Top-p Sampling  

All features are implemented to match the original model architecture.

## **Repository Structure**
- `model.py`: Contains the LLaMa 2 model implementation.  
- `inference.py`: Provides code for running inference using the model.

## **Testing with Pretrained Weights**
I have successfully integrated the LLaMa 2-7B model's weights for testing the inference.  
To obtain the model weights:
- Download directly from the [LLaMa website](https://ai.meta.com/llama/).  
- Use the `download.sh` script from the [official LLaMa repository](https://github.com/meta-llama/llama).

## **Running the Model (Inference Only)**

### **Requirements**
- `sentencepiece`  
- `torch`  
- `tqdm`

### **Steps**
1. Update the paths for the tokenizer and model weights in `inference.py`.  
2. Run the script:  
   ```bash
   python inference.py

## Resources and references used 
- Umar Jamil's LLaMa explanation: YouTube Link
- Official LLaMa 2 Repository: GitHub Link
- Various Medium blogs for understanding the model architecture.

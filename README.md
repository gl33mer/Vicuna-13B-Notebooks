# Context

Vicuna-13B is a new open-source chatbot developed by researchers from UC Berkeley, CMU, Stanford, and UC San Diego to address the lack of training and architecture details in existing large language models (LLMs) such as OpenAI's ChatGPT. This chatbot has been fine-tuned from a LLaMA base model using approximately 70,000 user-shared conversations collected from ShareGPT.com with public APIs, resulting in an enhanced dataset.

The preliminary evaluation of Vicuna-13B using GPT-4 as a judge shows that it achieves over 90% quality of OpenAI ChatGPT and Google Bard while outperforming other models like LLaMA and Stanford Alpaca in more than 90% of cases. The research team optimized Vicuna's performance with several key improvements, including memory optimizations, multi-round conversations, and cost reduction via Spot Instance.

To train Vicuna, the research team collected around 70,000 conversations from ShareGPT.com and enhanced the training scripts provided by Alpaca to better handle multi-round conversations and long sequences. The team used PyTorch FSDP on 8 A100 GPUs to train Vicuna in just one day. To serve the demo, the team implemented a lightweight distributed serving system capable of serving multiple models with distributed workers. This system supports flexible plug-in of GPU workers from both on-premise clusters and the cloud.

The evaluation framework proposed by the research team offers a promising approach to assessing chatbot performance in a consistent and automated manner. The team's use of diverse question categories and careful prompt engineering highlights the potential for this framework to uncover differences in chatbot performance that may not be easily discernible through human evaluation.

# Assets

This repository has the Jupyter Notebooks to get your instance of Vicuna up and running on Google Colab. We also have another notebook for training this model.

[For Running on Google Colab with a WebUI](Vicuna_13b_gpu_WebUI.ipynb)

[For Training on Google Colab](1508_Vicuna_13B_train.ipynb)

# Reduce GPU Memory

Reduce the batch size: One of the easiest ways to reduce GPU memory usage is by reducing the batch size. In the code you provided, you can try reducing the per_device_train_batch_size and per_device_eval_batch_size arguments to a smaller value. However, this may increase the training time since it takes more iterations to cover the entire dataset.

Use mixed precision training: Another way to reduce GPU memory usage is by using mixed precision training. This involves using lower precision floating-point numbers (e.g., float16 instead of float32) for some parts of the computation, which reduces memory usage and can speed up training. In the code you provided, you can try setting the bf16 and tf32 arguments to False to disable mixed precision training.

Use gradient accumulation: If reducing the batch size is not an option, you can try using gradient accumulation. This involves accumulating the gradients over multiple iterations before updating the model parameters, which reduces the memory usage per iteration. In the code you provided, you can try increasing the gradient_accumulation_steps argument to a larger value (e.g., 2, 4, etc.).

Use smaller models: If none of the above options work, you can try using a smaller model. In the code you provided, you can try using a smaller pre-trained model by setting the model_name_or_path argument to a smaller model (e.g., "distilgpt2" instead of "decapoda-research_llama-13b-hf"). However, this may reduce the model's performance on the task.


# Tips

This is an on-going project, you should change the configuration on "export CUDA_VISIBLE_DEVICES=0" and also tune the torchrun variables. Don't forget to manually change the .json to fix the problem from LlaMa to Llama.

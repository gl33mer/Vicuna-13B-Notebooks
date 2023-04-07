### Context

Vicuna-13B is a new open-source chatbot developed by researchers from UC Berkeley, CMU, Stanford, and UC San Diego to address the lack of training and architecture details in existing large language models (LLMs) such as OpenAI's ChatGPT. This chatbot has been fine-tuned from a LLaMA base model using approximately 70,000 user-shared conversations collected from ShareGPT.com with public APIs, resulting in an enhanced dataset.

The preliminary evaluation of Vicuna-13B using GPT-4 as a judge shows that it achieves over 90% quality of OpenAI ChatGPT and Google Bard while outperforming other models like LLaMA and Stanford Alpaca in more than 90% of cases. The research team optimized Vicuna's performance with several key improvements, including memory optimizations, multi-round conversations, and cost reduction via Spot Instance.

To train Vicuna, the research team collected around 70,000 conversations from ShareGPT.com and enhanced the training scripts provided by Alpaca to better handle multi-round conversations and long sequences. The team used PyTorch FSDP on 8 A100 GPUs to train Vicuna in just one day. To serve the demo, the team implemented a lightweight distributed serving system capable of serving multiple models with distributed workers. This system supports flexible plug-in of GPU workers from both on-premise clusters and the cloud.

The evaluation framework proposed by the research team offers a promising approach to assessing chatbot performance in a consistent and automated manner. The team's use of diverse question categories and careful prompt engineering highlights the potential for this framework to uncover differences in chatbot performance that may not be easily discernible through human evaluation.

# Assets

This repository has the Jupyter Notebooks to get your instance of Vicuna up and running on Google Colab. We also have another notebook for training this model.

[For Running on Google Colab with a WebUI](Vicuna_13b_gpu_WebUI.ipynb)

[For Training on Google Colab](1508_Vicuna_13B_train.ipynb)

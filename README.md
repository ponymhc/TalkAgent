# TalkAgent
## Demo
https://github.com/ponymhc/AudioAgent/assets/134651366/36a7c7d2-a8c8-4994-b207-f2db99536479
## 项目简介
这个项目实现了一个简单的语音对话机器人，能够进行智能问答和语音交互。该机器人利用语音和自然语言处理技术，包括语音识别、语音合成和生成式语言模型, 为用户提供了实时语音交互体验。此外，机器人还具备基于检索的问答和工具调用功能，提升了交互的智能性和实用性。项目由以下三个主要组件构成:
1. ASR: 语音识别系统, 用于将用户语音输入转写为文本作为聊天机器人的输入。
2. ChatBot: 基于LLM的对话机器人, 可智能地根据用户输入进行检索回答或工具调用功能。
3. TTS: 语音合成系统，对话系统的输出文本合成为普通话语音并可以与文本同时进行流式输出。
## 功能介绍
### ASR
语音识别系统是使用使用 fast-Whisper 部署的 whisper-small 模型，可在CPU上快速进行推理。
* Toolkit: ESPnet [[Repo](https://github.com/espnet/espnet)] [[Paper](https://arxiv.org/abs/1804.00015)]
* Model: Faster-Whisper [[Repo](https://github.com/SYSTRAN/faster-whisper)] [[Paper](https://arxiv.org/abs/2212.04356)]
### ChatBot
聊天机器人是基于 Llama-3-Chinese-Instruct 8 bit 量化版本构建的, 在此基础上使用 Langchain 构建了一个主动型 Agent, 可以根据用户输入进行自主决策采用的行为，包括二阶段的知识库检索增强生成，工具调用，以便更智能化地回答用户问题。   
* Tookit: Langchain [[Repo](https://github.com/langchain-ai/langchain)] [[Homepage](https://www.langchain.com/)]
* LLM: Llama-3-Chinese-Instruct [[Repo](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)]
* Embedding: gte [[Huggingface](https://huggingface.co/thenlper/gte-large-zh)]
* Reranker: bge [[Huggingface](https://huggingface.co/BAAI/bge-reranker-base)]
* Inference framework: Llama.cpp [[Repo](https://github.com/ggerganov/llama.cpp)]
#### 二阶段检索增强生成
![Two stage RAG](https://github.com/ponymhc/AudioAgent/blob/main/image/two_stage_rag.png)
### TTS
语音合成系统采用是使用ESPnet工具包训练的端到端的语音合成模型, 模型架构采用的是 VITS 端到端地进行文本到语音的合成。
* Toolkit: ESPnet [[Repo](https://github.com/espnet/espnet)] [[Paper](https://arxiv.org/abs/1804.00015)]
* Dataset: CSMSC [[Homepage](https://www.data-baker.com/open_source.html)]
* Model: VITS [[Repo](https://github.com/jaywalnut310/vits)] [[Paper](https://arxiv.org/abs/2106.06103)]
#### 训练日志
![Discriminator_loss](https://github.com/ponymhc/AudioAgent/blob/main/image/vits_discriminator_loss.png)
![Generator_loss](https://github.com/ponymhc/AudioAgent/blob/main/image/vits_generator_loss.png)
![Generator_mel_loss](https://github.com/ponymhc/AudioAgent/blob/main/image/vits_generator_mel_loss.png)
## 使用
语言模型使用了Llama.cpp进行部署推理，请先安装llama-cpp-python，强烈建议使用CUDA编译版本。
```
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
安装运行所需依赖项。
```
pip install -r requirements.txt
```
现在，只需运行run.sh即可！！！
```
sh run.sh
```

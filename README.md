# AudioLlama
## 项目简介
这个项目实现了一个简单的语音对话机器人，能够进行自然语言理解和语音交互。该机器人利用语音处理技术和大语言模型，包括语音识别、语音合成和主动型Agent, 为用户提供了实时语音交互体验。此外，机器人还具备基于检索的问答和工具调用功能，提升了交互的智能性和实用性。项目由以下三个主要组件构成:
1. ASR: 语音识别系统, 用于将用户语音输入转写为文本作为聊天机器人的输入。
2. ChatBot: 基于LLM的对话机器人, 可智能地根据用户输入进行选择闲聊、检索回答或工具调用功能。
3. TTS: 语音合成系统，对话系统的输出文本合成为普通话语音并可以与文本同时进行流式输出。

## 功能介绍
### ASR
语音识别系统是使用 ESPnet 开源工具包训练的一个端到端语音识别模型。使用 LoRA 在中文语音转写数据集上微调的 Whisper 模型实现较高精度的普通话语音转写。
* Toolkit: ESPnet [[Repo](https://github.com/espnet/espnet)] [[Paper](https://arxiv.org/abs/1804.00015)]
* Dataset: AIShell [[Homepage](https://www.openslr.org/33/)] [[Paper](https://arxiv.org/abs/1709.05522)]
* Model: Whisper [[Repo](https://github.com/openai/whisper)] [[Paper](https://arxiv.org/abs/2212.04356)]

### ChatBot
聊天机器人是基于 Llama-3-Chinese-Instruct 8 bit 量化版本构建的, 在此基础上使用 Langchain 构建了一个主动型 Agent, 可以根据用户输入进行自主决策采用的行为，包括二阶段的知识库检索增强生成，工具调用，以便更智能化地回答用户问题。   
* Tookit: Langchain [[Repo](https://github.com/langchain-ai/langchain)] [[Homepage](https://www.langchain.com/)]
* LLM: Llama-3-Chinese-Instruct [[Repo](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)]
* Embedding:
* Reranker
* Inference framework: Llama.cpp [[Repo](https://github.com/ggerganov/llama.cpp)]
二阶段检索增强生成
![Two stage RAG](https://github.com/ponymhc/AudioLlama/blob/main/image/two_stage_retrieval.jpg)
### TTS
语音合成系统采用是使用ESPnet工具包训练的端到端的语音合成模型, 模型架构采用的是 FastSpeech2 + HifiGAN, 使用的是 JETS 训练方法从而避免了传统二阶段语音合成系统中声学模型和声码器之间 mel 谱mismatch的问题。
* Toolkit: ESPnet [[Repo](https://github.com/espnet/espnet)] [[Paper](https://arxiv.org/abs/1804.00015)]
* Dataset: CSMSC [[Homepage](https://www.data-baker.com/open_source.html)]
* Model: FastSpeech2 + HifiGAN
  * FastSpeech2 [[Paper](https://arxiv.org/abs/2006.04558)]
  * HifiGAN [[Repo](https://github.com/jik876/hifi-gan)] [[Paper](https://arxiv.org/abs/2010.05646)]
* Training Method: JETS [[Repo](https://github.com/imdanboy/jets)] [[Paper](https://arxiv.org/abs/2203.16852)]

## 使用
模型使用了Llama.cpp进行部署推理，在使用前请确保已经完成Llama.cpp的编译。
```
git clone git@github.com:ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DLLAMA_CUDA=ON
cmake --build build --config Release
```
安装运行所需依赖项。
```
pip install -r requirements.txt
```
现在，只需运行run.sh即可！！！
```
sh run.sh
```

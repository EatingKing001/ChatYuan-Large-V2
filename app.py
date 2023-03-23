import os
import gradio as gr
import clueai
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v2")
model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2").half()
# 使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def preprocess(text):
  text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  return text.replace("\\n", "\n").replace("\\t", "\t").replace('%20','  ')

def answer(text, sample=True, top_p=1, temperature=0.7):
  '''sample：是否抽样。生成任务，可以设置为True;
  top_p：0-1之间，生成的内容越多样'''
  text = preprocess(text)
  encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device) 
  if not sample:
    out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
  else:
    out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
  out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
  return postprocess(out_text[0])

def clear_session():
    return '', None

def chatyuan_bot(input, history):
    history = history or []
    if len(history) > 5:
       history = history[-5:]

    context = "\n".join([f"用户：{input_text}\n小元：{answer_text}" for input_text, answer_text in history])
    print(context)

    input_text = context + "\n用户：" + input + "\n小元："
    output_text = answer(input_text)
    history.append((input, output_text))
    print(history)
    return history, history

block = gr.Blocks()

with block as demo:
    gr.Markdown("""<h1><center>元语智能——ChatYuan</center></h1>
        <font size=4>回答来自ChatYuan, 是模型生成的结果, 请谨慎辨别和参考, 不代表任何人观点</font>

    """)
    chatbot = gr.Chatbot(label='ChatYuan')
    message = gr.Textbox()
    state = gr.State()
    message.submit(chatyuan_bot, inputs=[message, state], outputs=[chatbot, state])
    with gr.Row():
      clear_history = gr.Button("👋 清除历史对话")
      clear = gr.Button('🧹 清除发送框')
      send = gr.Button("🚀 发送")
      
    send.click(chatyuan_bot, inputs=[message, state], outputs=[chatbot, state])
    clear.click(lambda: None, None, message, queue=False)
    clear_history.click(fn=clear_session , inputs=[], outputs=[chatbot, state], queue=False)
    

def ChatYuan(api_key, text_prompt):

    cl = clueai.Client(api_key,
                        check_api_key=True)
    # generate a prediction for a prompt
    # 需要返回得分的话，指定return_likelihoods="GENERATION"
    prediction = cl.generate(model_name='ChatYuan-large', prompt=text_prompt)
    # print the predicted text
    print('prediction: {}'.format(prediction.generations[0].text))
    response = prediction.generations[0].text
    if response == '':
        response = "很抱歉，我无法回答这个问题"

    return response
  
def chatyuan_bot_api(api_key, input, history):
    history = history or []

    if len(history) > 5:
      history = history[-5:]

    context = "\n".join([f"用户：{input_text}\n小元：{answer_text}" for input_text, answer_text in history])
    print(context)

    input_text = context + "\n用户：" + input + "\n小元："
    output_text = ChatYuan(api_key, input_text)
    history.append((input, output_text))
    print(history)
    return history, history

block = gr.Blocks()

with block as demo_1:
    gr.Markdown("""<h1><center>元语智能——ChatYuan</center></h1>
    <font size=4>回答来自ChatYuan, 以上是模型生成的结果, 请谨慎辨别和参考, 不代表任何人观点</font>
    
    <font size=4>在使用此功能前，你需要有个API key. API key 可以通过这个<a href='https://www.clueai.cn/' target="_blank">平台</a>获取</font>
    """)
    api_key = gr.inputs.Textbox(label="请输入你的api-key(必填)", default="", type='password')
    chatbot = gr.Chatbot(label='ChatYuan')
    message = gr.Textbox()
    state = gr.State()
    message.submit(chatyuan_bot_api, inputs=[api_key,message, state], outputs=[chatbot, state])
    with gr.Row():
      clear_history = gr.Button("👋 清除历史对话")
      clear = gr.Button('🧹 清除发送框')
      send = gr.Button("🚀 发送")

    send.click(chatyuan_bot_api, inputs=[api_key,message, state], outputs=[chatbot, state])
    clear.click(lambda: None, None, message, queue=False)
    clear_history.click(fn=clear_session , inputs=[], outputs=[chatbot, state], queue=False)

block = gr.Blocks()
with block as introduction:
    gr.Markdown("""<h1><center>元语智能——ChatYuan</center></h1>
    
<font size=4>😉ChatYuan: 元语功能型对话大模型
<br>
<br>
👏ChatYuan-large-v2是一个支持中英双语的功能型对话语言大模型，是继ChatYuan系列中ChatYuan-large-v1开源后的又一个开源模型。ChatYuan-large-v2使用了和 v1版本相同的技术方案，在微调数据、人类反馈强化学习、思维链等方面进行了优化。

ChatYuan-large-v2是ChatYuan系列中以轻量化实现高质量效果的模型之一，用户可以在消费级显卡、 PC甚至手机上进行推理（INT4 最低只需 400M ）。

在chatyuan-large-v1的原有功能的基础上，我们给模型进行了如下优化：
- 增强了基础能力。原有上下文问答、创意性写作能力明显提升。
- 新增了拒答能力。对于一些危险、有害的问题，学会了拒答处理。
- 新增了代码生成功能。对于基础代码生成进行了一定程度优化。
- 新增了表格生成功能。使生成的表格内容和格式更适配。
- 增强了基础数学运算能力。
- 最大长度token数扩展到4096。
- 增强了模拟情景能力。
- 新增了中英双语对话能力。.<br>
<br>
👀<a href='https://www.cluebenchmarks.com/clueai.html'>PromptCLUE-large</a>在1000亿token中文语料上预训练, 累计学习1.5万亿中文token, 并且在数百种任务上进行Prompt任务式训练. 针对理解类任务, 如分类、情感分析、抽取等, 可以自定义标签体系; 针对多种生成任务, 可以进行采样自由生成.  <br> 
<br>
🚀<a href='https://www.clueai.cn/chat' target="_blank">在线Demo</a> &nbsp; | &nbsp; <a href='https://modelscope.cn/models/ClueAI/ChatYuan-large/summary' target="_blank">ModelScope</a> &nbsp; | &nbsp; <a href='https://huggingface.co/ClueAI/ChatYuan-large-v1' target="_blank">Huggingface</a> &nbsp; | &nbsp; <a href='https://www.clueai.cn' target="_blank">官网体验场</a> &nbsp; | &nbsp; <a href='https://github.com/clue-ai/clueai-python#ChatYuan%E5%8A%9F%E8%83%BD%E5%AF%B9%E8%AF%9D' target="_blank">ChatYuan-API</a> &nbsp; | &nbsp; <a href='https://github.com/clue-ai/ChatYuan' target="_blank">Github项目地址</a> &nbsp; | &nbsp; <a href='https://openi.pcl.ac.cn/ChatYuan/ChatYuan/src/branch/main/Fine_tuning_ChatYuan_large_with_pCLUE.ipynb' target="_blank">OpenI免费试用</a> &nbsp;
</font>
    """)


gui = gr.TabbedInterface(interface_list=[introduction,demo, demo_1], tab_names=["相关介绍","开源模型", "API调用"])
gui.launch(quiet=True,show_api=False, share = False)
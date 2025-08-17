import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('/root/.cache/modelscope/hub/models/iic/CosyVoice2-0___5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False, bf16=True)

def get_module_dtype(module):
    for p in module.parameters():
        return p.dtype
    # fallback: 看子模块
    for child in module.children():
        dtype = get_module_dtype(child)
        if dtype is not None:
            return dtype
    return None

print(f"[INFO] LLM dtype: {get_module_dtype(cosyvoice.model.llm)}")
print(f"[INFO] Flow dtype: {get_module_dtype(cosyvoice.model.flow)}")

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# bistream usage, you can use generator as input, this is useful when using text llm model as input
# NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
def text_generator():
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'
for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
import queue, torch, os, warnings, time
from yarl import Query
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")

def _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=224, video_framerate=1, s=None, e=None):
    # speed up video decode via decord.

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
        slice_len = patch_images.shape[0]

        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))

def inference(
        video_path, query, model, tokenizer, 
        image_processor, conv_mode="simple", 
        max_frames=100, video_framerate=1, temperature=0.2, 
        top_p=None, num_beams=1
    ):
    if video_path is not None:
        video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor, max_frames=max_frames, video_framerate=video_framerate)

        cur_prompt = query
        if model.config.mm_use_im_start_end:
            query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            query = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + query

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        start_time = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video_frames.half().cuda(),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        output_ids = output_ids.sequences
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print("=" * 80)
        print("Time: ", time.time() - start_time)
        print("=" * 80)
        print(f"Query: {cur_prompt}")
        print("=" * 80)
        print(outputs)
        print("*" * 110, "\n")
        return outputs

if __name__ == '__main__':
    # Model Parameter
    model_path = "Chat-UniVi/Chat-UniVi"  # or "Chat-UniVi/Chat-UniVi-13B"、"Chat-UniVi/Chat-UniVi-v1.5"
    prefix = "/home/weiji/yqin/data/vmr/nextVMR/video224"
    video_path = os.path.join(prefix, "9996338863.mp4")

    # The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
    # When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
    max_frames = 100

    # The number of frames retained per second in the video.
    video_framerate = 1

    # Input Text
    # qs = "Describe the video."
    query = "The adult is playing the guitar while a woman sings on stage with a drum set and a guitar behind her."
    # query = "The baby is sitting on the bed and reading a book."
    # query = "There is a pig in the video."
    qs = "This is a 60.6299-second video clip. Provide a single time interval within the range [0, 60.6299], formatted as ‘[a, b]’, that corresponds to the segment of the video best matching the query: {query} Respond with only the numeric interval.".format(query=query)
    # qs = "This is a 60.62-second video. Provide up to 3 time intervals within [0, 60.62], formatted as ‘[a, b]’, ‘[a, b], [c, d]’, or ‘[a, b], [c, d], [e, f]’, that best match the query: {query}. Respond only with the numeric intervals.".format(query=query)
    # qs = "This is a 60.62-second video clip for the query: {query}\n\nStep 1: If the query does not exist in the video, respond with ‘no’.\nStep 2: If the query exists, respond with ‘yes’ followed by a single time interval within the range [0, 60.62], formatted as ‘[a, b]’, that corresponds to the segment of the video best matching the query. Only provide the response in the format: ‘no’ or ‘yes [a, b]’.".format(query=query)
 
    # Sampling Parameter
    conv_mode = "simple"
    temperature = 0.2
    top_p = None
    num_beams = 1

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = "ChatUniVi"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    if model.config.config["use_cluster"]:
        for n, m in model.named_modules():
            m = m.to(dtype=torch.bfloat16)
            
    # Run inference
    inference(video_path, qs, model, tokenizer, image_processor, conv_mode, max_frames=100, video_framerate=1, temperature=0.2, top_p=None, num_beams=1)
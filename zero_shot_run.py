from tqdm import tqdm
from zero_shot_vmr import *
from zero_shot_stat import *
import warnings, os
import argparse
os.environ["DEEPSPEED_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore")

def initialize_model(model_path="Chat-UniVi/Chat-UniVi"):
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
        for _, m in model.named_modules():
            m = m.to(dtype=torch.bfloat16)
    
    return tokenizer, model, image_processor, context_len

def cal_iou(a, b):
    if a =='nan' or b == 'nan':
        return 'nan'
    else:
        a_start, a_end = a
        b_start, b_end = b

        # Calculate the intersection
        intersection_start = max(a_start, b_start)
        intersection_end = min(a_end, b_end)
        intersection = max(0, intersection_end - intersection_start)
        # Calculate the union
        union_start = min(a_start, b_start)
        union_end = max(a_end, b_end)
        union = union_end - union_start

        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        return iou

def prompt(duration, query):
    return (
        "This is a {duration:.2f} second video clip. "
        "Provide a single time interval within the range [0, {duration:.2f}], "
        "formatted as ‘[a, b]’, "
        "that corresponds to the segment of the video best matching the query: {query} "
        "Respond with only the numeric interval."
    ).format(duration=duration, query=query)
    
def promptno(duration, query):
    return (
        "This is a {duration:.2f} second video clip. "
        "Step 1: Does the query: '{query}' correspond to any segment in the video? "
        "Respond with 'yes' or 'no' on the first line. "
        "Step 2: If your response in Step 1 is 'yes', provide only one time interval formatted as '[a, b]' "
        "on the second line, where 'a' and 'b' are numbers such that 0 ≤ a < b ≤ {duration:.2f}. "
        "If your response in Step 1 is 'no', leave the second line blank. "
        "Do not include any additional explanation or text beyond the specified format."
    ).format(duration=duration, query=query)
    
def append_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + '\n')
        
def sanity_check(prev_result): #load previous result to get qid dict
    if not os.path.exists(prev_result):
        return {}
    else:
        prev = load_jsonl(prev_result)
        qid_dict = {}
        for p in prev:
            qid_dict[p["qid"]] = p["vid"]
        return qid_dict
        
if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=2 python zero_shot_run.py --task test 2> /dev/null | tee test.log
    # CUDA_VISIBLE_DEVICES=2 python zero_shot_run.py --task val 2> /dev/null | tee val.log
    # CUDA_VISIBLE_DEVICES=2 python zero_shot_run.py --task test2 | tee test2.log
    # CUDA_VISIBLE_DEVICES=2 python zero_shot_run.py --task val2 | tee val2.log
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="test", help="test or val")
    args = parser.parse_args()
    
    # Step 1: Set path and parameters, and load model
    ## model Parameter
    model_path = "Chat-UniVi/Chat-UniVi"
    model_path = os.path.expanduser(model_path)
    tokenizer, model, image_processor, context_len = initialize_model(model_path=model_path)
    
    # Step 2: Set video parameters and inference
    ## task path
    prefix = "/home/weiji/yqin/data/vmr/nextVMR/video224"
    ## video Parameter
    conv_mode = "simple"
    max_frames = 100 # 100 frames per video
    video_framerate = 1 # 1 frame per second
    ## prepare two tasks
    if args.task == "test":
        task = load_jsonl("single_test.jsonl")
    elif args.task == "val":
        task = load_jsonl("single_val.jsonl")
    elif args.task == "test2":
        task = load_jsonl("singleno_test.jsonl")
    elif args.task == "val2":
        task = load_jsonl("singleno_val.jsonl")
        
    ## single query
    # vid = "9996338863"
    # duration = 60.6299
    # query = "The adult is playing the guitar while a woman sings on stage with a drum set and a guitar behind her."
    # # query = "There are 100 people in the video."
    # video_path = os.path.join(prefix, f"{vid}.mp4")
    # qs_1 = promptno(duration, query)
    # outputs_1 = inference(video_path, qs, model, tokenizer, image_processor, conv_mode, max_frames, video_framerate)
    # signal = True if "yes" in outputs_1[:5].lower() else False
    # if signal:
    #     qs_2 = prompt(duration, query)
    #     outputs_2 = inference(video_path, qs_2, model, tokenizer, image_processor, conv_mode, max_frames, video_framerate)
    #     outputs_2 = [float(i) for i in outputs_2.strip("[]").split(", ")]
    #     iou = cal_iou(outputs_2, [0, 60.6299])
    #     print(f"Task 1: {iou}")
    # else:
    #     print(f"Task 1: no iou")
    
    # Step 3: Run inference over task, write single result to jsonl
    result_file = f"result_{args.task}.jsonl"
    qid_dict = sanity_check(result_file)
    for i, t in tqdm(enumerate(task)):
        if t["qid"] in qid_dict and t["vid"] == qid_dict[t["qid"]]:
            continue
        print(f"Start from task {i+1}/{len(task)}, there remain {len(task)-i-1} tasks.")
        vid = t["vid"]
        duration = t["duration"]
        query = t["query"]
        qid = t["qid"]
        if t["gt_timestamps"] == []:
            gt = "nan"
        else:     
            gt = t["gt_timestamps"][0]
            gt = [gt[0]*duration, gt[1]*duration]
        qs_1 = promptno(duration, query)
        video_path = os.path.join(prefix, f"{vid}.mp4")
        retry_count = 0
        while retry_count < 3:
            try:
                outputs_1 = inference(video_path, qs_1, model, tokenizer, image_processor, conv_mode, max_frames, video_framerate)
                signal = True if "yes" in outputs_1[:5].lower() else False
                if signal:
                    qs_2 = prompt(duration, query)
                    outputs_2 = inference(video_path, qs_2, model, tokenizer, image_processor, conv_mode, max_frames, video_framerate)    
                    outputs_2 = [float(i) for i in outputs_2.strip("[]").split(", ")]
                    iou = cal_iou(outputs_2, gt)
                    print(f"IOU: {iou}")
                    single_result = {
                        "vid": vid,
                        "qid": qid,
                        "outputs": outputs_2,
                        "gt": gt,
                        "iou": iou,
                    }
                else:
                    single_result = {
                        "vid": vid,
                        "qid": qid,
                        "outputs": [],
                        "gt": gt,
                        "iou": "nan",
                    }
                    print("IOU: nan")
                append_to_jsonl(result_file, single_result)
                break
            except:
                retry_count += 1
                print(f"Task {i+1}/{len(task)} failed. Retry {retry_count}/3.")
                if retry_count == 3:
                    # record the failed vid and qid into one line in txt
                    with open(f"failed_{args.task}.txt", 'a') as f:
                        f.write(f"{vid} {qid}\n")
    print(f"Task {args.task} finished.")
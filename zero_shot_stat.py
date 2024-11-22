import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_data(data, save_to=None):
    single_data = []
    singleno_data = []
    stat_data = {}
    for item in data:
        interval = item['gt_timestamps']
        if len(interval) == 1:
            stat_data['single'] = stat_data.get('single', 0) + 1
            single_data.append(item)
            singleno_data.append(item)
        elif len(interval) == 0:
            stat_data['empty'] = stat_data.get('empty', 0) + 1
            singleno_data.append(item)
        else:
            stat_data['multi'] = stat_data.get('multi', 0) + 1
    stat_data = dict(sorted(stat_data.items(), key=lambda x: x[0]))
    if save_to:  # save single_data to jsonl file
        with open(save_to, 'w') as f:
            for item in single_data:
                f.write(json.dumps(item) + '\n')
        with open(save_to.replace("single", "singleno"), 'w') as f:
            for item in singleno_data:
                f.write(json.dumps(item) + '\n')
    return stat_data, single_data, singleno_data

if __name__ == '__main__':
    test_path = "/home/weiji/yqin/data/vmr/nextVMR/annos/test_v3.jsonl"
    val_path = "/home/weiji/yqin/data/vmr/nextVMR/annos/val_v3.jsonl"

    test_data = load_jsonl(test_path)
    val_data = load_jsonl(val_path)

    stat_val, single_val, singleno_val = analyze_data(val_data, save_to="single_val.jsonl")
    stat_test, single_test, singleno_test = analyze_data(test_data, save_to="single_test.jsonl")
    
    print(f"Val: {stat_val}")
    print(f"Test: {stat_test}")
    
    """
    Val: {'empty': 2340, 'multi': 1007, 'single': 8461}
    Test: {'empty': 6448, 'multi': 2769, 'single': 24577}
    """
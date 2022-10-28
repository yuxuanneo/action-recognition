import torch

from mmaction.apis import init_recognizer, inference_recognizer

def main():
    config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

    # assign the desired device.
    device = 'cpu' # or 'cpu'
    device = torch.device(device)

    # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, checkpoint_file, device=device)

    # test a single video and show the result:
    video = 'demo/demo.mp4'
    labels = 'tools/data/kinetics/label_map_k400.txt'
    results = inference_recognizer(model, video)

    # show the results
    labels = open('tools/data/kinetics/label_map_k400.txt').readlines()
    labels = [x.strip() for x in labels]
    results = [(labels[k[0]], k[1]) for k in results]

    print(f'The top-5 labels with corresponding scores are:')
    for result in results:
        print(f'{result[0]}: ', result[1])

if __name__ == "__main__":
    main()

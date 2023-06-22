import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSP
from effortless_config import Config
from os import path
from preprocess import Dataset
from tqdm import tqdm
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from einops import rearrange
from ddsp.utils import get_scheduler
import numpy as np


class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "../drive/MyDrive/runs_resnet_softmax"
    STEPS = 500000
    BATCH = 16
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000
    LOAD_FROM_CHECKPOINT = False
    CHECKPOINT = "../drive/MyDrive/runs_resnet_1/debug/state6609.713998248192556.pth"

def main():

    args.parse_args()

    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDSP(**config["model"]).to(device)

    if args.LOAD_FROM_CHECKPOINT:
        model.load_state_dict(torch.load(args.CHECKPOINT))

    dataset = Dataset(config["preprocess"]["out_dir"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        args.BATCH,
        True,
        drop_last=True,
    )

    mean_loudness, std_loudness = mean_std_loudness(dataloader)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

    writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

    with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
        yaml.safe_dump(config, out_config)

    opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

    schedule = get_scheduler(
        len(dataloader),
        args.START_LR,
        args.STOP_LR,
        args.DECAY_OVER,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

    best_loss = float("inf")
    mean_loss = 0
    n_element = 0
    step = 0
    epochs = int(np.ceil(args.STEPS / len(dataloader)))

    model.train()

    for e in tqdm(range(epochs)):
        for s, p, l in dataloader:
            s = s.to(device)
            if len(s.shape) == 1:
                s = s.unsqueeze(0)
            p = p.unsqueeze(-1).to(device)
            l = l.unsqueeze(-1).to(device)

            #l = (l - mean_loudness) / std_loudness this seems to produce worse results than no standardization
            #p = 

            y, pitch_loss = model(s, p, l)
            y = y.squeeze(-1)
            print(pitch_loss)
            y = y[:, :s.shape[-1]]
            ori_stft = multiscale_fft(
                s,
                config["train"]["scales"],
                config["train"]["overlap"],
            )
            rec_stft = multiscale_fft(
                y,
                config["train"]["scales"],
                config["train"]["overlap"],
            )

            loss = 0.0001*pitch_loss
            for s_x, s_y in zip(ori_stft, rec_stft):
                lin_loss = (s_x - s_y).abs().mean()
                log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
                loss = loss + lin_loss + log_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            

            writer.add_scalar("loss", loss.item(), step)

            step += 1

            n_element += 1
            mean_loss += (loss.item() - mean_loss) / n_element

        if not e % 10 or e == epochs-1:
            writer.add_scalar("lr", schedule(e), e)
            writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
            writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)
            scheduler.step()
            if mean_loss < best_loss:
                print(mean_loss)
                best_loss = mean_loss
                torch.save(
                    model.state_dict(),
                    path.join(args.ROOT, args.NAME, f"state{e}{mean_loss}.pth"),
                )

            mean_loss = 0
            n_element = 0

            audio = torch.cat([s, y], -1).reshape(-1).detach().cpu().numpy()

            sf.write(
                path.join(args.ROOT, args.NAME, f"eval_{e:06d}.wav"),
                audio,
                config["preprocess"]["sampling_rate"],
            )

if __name__ == "__main__":
    main()
    
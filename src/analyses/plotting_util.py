import os

from clat.intan.channels import Channel


def extract_target_channel_data(channel: Channel, data):
    """Add a SpikeTimes_{channel} column extracted from the SpikeTimes dict."""
    out = data.copy()
    out[f"SpikeTimes_{channel.value}"] = data["SpikeTimes"].apply(
        lambda x: x[next(filter(lambda k: k.value == channel.value, x.keys()), None)]
    )
    return out

def raster_save_path(raster_type, date, round_no, channel):
    date_fmt = date.replace("-", "")[2:]
    folder = f"{date_fmt}_round{round_no}_new"
    save_dir = "/mixed_manual_raster_plots/"
    return os.path.join(save_dir, folder, f"{date}_round{round_no}_{channel}.png")
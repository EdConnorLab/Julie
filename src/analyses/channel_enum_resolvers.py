from clat.intan.channels import Channel


def get_value_from_dict_with_channel(channel, dictionary):
    if isinstance(channel, str):
        return dictionary[channel]
    else:
        for key, value in dictionary.items():
            if key.value == channel.value:
                return value


def is_channel_in_dict(channel, diction):
    if isinstance(channel, str):
        if channel in list(diction.keys()):
            return True
    else:
        for key in diction:
            if channel.value == key.value:
                return True


def convert_to_enum(channel_str):
    enum_name = channel_str.split('.')[1]
    return getattr(Channel, enum_name)


def drop_duplicate_channels(raw_data, sorted_data):
    channels_with_units = (sorted_data['SpikeTimes'][0].keys())
    sorted_channels = [int(s.split('_')[1][1:]) for s in channels_with_units]
    sorted_enum_channels = list(set([Channel(f'C-{channel:03}') for channel in sorted_channels]))
    # remove channel only if it exists as index
    for channel in sorted_enum_channels:
        if channel in raw_data.index:
            raw_data = raw_data.drop(channel)
    return raw_data


def drop_duplicate_channels_with_matching_time_window(df):
    """
    Parameters:
    df (pd DataFrame): columns containing 'Time Window' and 'Cell'

    """
    df['Cell'] = df['Cell'].astype(str)
    df['BaseChannel'] = df['Cell'].apply(lambda x: x.split('_Unit')[0] if 'Unit' in x else x)
    mask = df.duplicated(subset=['BaseChannel', 'Time Window'], keep=False) & ~df['Cell'].str.contains('Unit')
    filtered_df = df[~mask].drop(columns='BaseChannel')

    return filtered_df

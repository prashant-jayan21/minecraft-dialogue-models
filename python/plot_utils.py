import re, csv, itertools, math, matplotlib.pyplot as plt
from statistics import mean, median, mode, stdev

def parse_losses(lines, train_or_val):
    """
    Args:
        lines: All lines of the train log
        train_or_val: Indicates whether train losses or validation losses are desired -- "train" or "val"

    Returns:
        List of losses ordered by epoch
    """
    if train_or_val == "train":
        loss_regex = re.compile(r"Training stats \| Loss: ([\d.]+)")
    elif train_or_val == "val":
        loss_regex = re.compile(r"Validation stats \| Loss: ([\d.]+)")

    losses = list(
        map(
            lambda x: float(x.group(1)),
            filter(
                lambda x: x != None,
                map(
                    loss_regex.search,
                    lines
                )
            )
        )
    )

    return losses

def plot_losses(train_log_file):
    """
    Args:
        train_log_file: A train log file name

    Returns:
        Nothing.
        Writes a file with validation losses and a file with the plot.
        Writes a file with avg train losses and a file with the plot.
    """
    with open(train_log_file) as f:
        all_lines = list(map(lambda t: t.strip(), f.readlines()))

    # get loss values

    train_losses = parse_losses(all_lines, "train")
    val_losses = parse_losses(all_lines, "val")

    # dump loss data to file

    train_losses_file = train_log_file[:-4] + ".train"
    val_losses_file = train_log_file[:-4] + ".val"

    with open(train_losses_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['epoch', 'train loss'])
        writer.writeheader()
        for i, train_loss in enumerate(train_losses):
            writer.writerow(
                {
                    "epoch": i + 1,
                    "train loss": train_loss
                }
            )
        print("DONE WRITING " + train_losses_file)

    with open(val_losses_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['epoch', 'validation loss'])
        writer.writeheader()
        for i, val_loss in enumerate(val_losses):
            writer.writerow(
                {
                    "epoch": i + 1,
                    "validation loss": val_loss
                }
            )
        print("DONE WRITING " + val_losses_file)

    # plot loss data and write to file

    plot_file = train_log_file[:-4] + ".train.png"

    plt.plot(train_losses)
    plt.savefig(plot_file)
    plt.gcf().clear()
    print("DONE WRITING " + plot_file)

    plot_file = train_log_file[:-4] + ".val.png"

    plt.plot(val_losses)
    plt.savefig(plot_file)
    plt.gcf().clear()
    print("DONE WRITING " + plot_file)

def plot_histogram(data, filename):
    print("Min:", min(data))
    print("Max:", max(data))
    print("Mean:", mean(data))
    print("Stdev:", stdev(data))

    # print([x for x in range(math.floor(min(data)), math.ceil(max(data)) + 1)])
    plt.hist(
        data,
        bins = [x for x in range(math.floor(min(data)), math.ceil(max(data)) + 1)],
        density=True
    )
    
    plt.xticks([x for x in range(math.floor(min(data)), math.ceil(max(data)) + 1)])
    plt.savefig(filename)
    plt.close()
    # plt.show()

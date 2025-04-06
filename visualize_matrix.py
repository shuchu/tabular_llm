import sys
import torch
import matplotlib.pyplot as plt
import click
import pickle


@click.group()
def cli():
    pass


@click.command()
@click.option("--input", required=True, help="The input tensor")
def heatmap(input):
    """ Visualize the tensor
    """
    data = {}

    with open(input, 'rb') as f:
        data = pickle.load(f)

    tensor = data["similarity_matrix"]
    tensor = tensor.numpy()
    plt.imshow(tensor, cmap='hot', interpolation='nearest')
    plt.savefig("visualize_matrix.png")

    #debug
    print(tensor)
    print(data["train_labels"])


@click.command()
@click.option("--input", help="The input tensor")
def heatmap_seaborn(input):
    """ Visualize the tensor using seaborn
    """
    import seaborn as sns
    
    # load the input data
    with open(input, 'rb') as f:
        data = pickle.load(f)

    tensor = data["similarity_matrix"]
    sns.set_theme(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(tensor, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    #sns.heatmap(tensor)
    plt.savefig("visualize_matrix_seaborn.png")


@click.command()
@click.option("--input", help="The input data pickle file")
@click.option("--label", help="The labels of the data")
@click.option("--save", default=False, help="Save the tsne result")
def tsne(input, label, save):
    """ Transform the tensor using tsne
    """
    from sklearn.manifold import TSNE
    import numpy as np
    import seaborn as sns

    # load the input data
    with open(input, 'rb') as f:
        data = pickle.load(f)

    # load the training labels
    tr_labels = data["train_labels"]

    sorted_idx = sorted(range(len(tr_labels)), key=lambda k: tr_labels[k])

    # calculate the tsne embedding
    tensor = data["train_embedding"]

    #debug
    print(tensor[0])
    print(tensor[1])

    tensor = tensor[sorted_idx]
    x_embedded = TSNE(n_components=2).fit_transform(tensor)
    
    # Make sure that the number of samples equals to the number of labels
    if len(tr_labels) != x_embedded.shape[0]:
        print(f"The number of labels: {len(tr_labels)} is not equal to the number of samples: {x_embedded.shape[0]}")
        return
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x = x_embedded[:,0], y = x_embedded[:,1], hue=tr_labels, palette="deep")

    plt.savefig("tsne.png")
        

# Add all commands
cli.add_command(heatmap)
cli.add_command(heatmap_seaborn)    
cli.add_command(tsne)

        
if __name__ == "__main__":
    cli()
    

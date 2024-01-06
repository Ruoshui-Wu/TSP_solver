import os
import numpy as np
import torch
from utils import load_model
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def make_oracle(model, xy, temperature=1.0):
    num_nodes = len(xy)

    xyt = torch.tensor(xy).float()[None]  # Add batch dimension

    with torch.no_grad():  # Inference only
        embeddings, _ = model.embedder(model._init_embed(xyt))

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = model._precompute(embeddings)

    def oracle(tour):
        with torch.no_grad():  # Inference only
            # Input tour with 0 based indices
            # Output vector with probabilities for locations not in tour
            tour = torch.tensor(tour).long()

            if len(tour) == 0:
                step_context = model.W_placeholder
            else:
                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

            # Compute query = context node embedding, add batch and step dimensions (both 1)
            query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

            # Create the mask and convert to bool depending on PyTorch version
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension

            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            p = torch.softmax(log_p / temperature, -1)[0, 0]
            assert (p[tour] == 0).all()
            assert (p.sum() - 1).abs() < 1e-5
            # assert np.allclose(p.sum().item(), 1)
        return p.numpy()

    return oracle

# TSP_Drawer can draw the solution step by step, which is given by trained neural network.
class TSP_Drawer:
    def __init__(self, xy, model):
        self.xy = xy
        self.sample = False
        self.tour = []
        self.total_cost = 0
        self.model = model
        self.oracle = make_oracle(model, xy) #process map using pretrained model
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.scatter = self.ax.scatter(self.xy[:, 0], self.xy[:, 1], s=40, color='blue')
        self.button_ax = plt.axes([0.7, 0.05, 0.1, 0.05])  # [left, bottom, width, height]
        self.button = Button(self.button_ax, 'next step')
        self.button.on_clicked(self.on_button_click)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

    def on_mouse_click(self, event):
        if event.button == 3:  # right mouse button click
            print('right mouse button click')
            new_point = np.array([[event.xdata, event.ydata]])
            self.xy = np.concatenate([self.xy, new_point], axis=0)
            self.oracle = make_oracle(self.model, self.xy)
            self.scatter.set_offsets(self.xy)
            self.fig.canvas.draw()

    def on_button_click(self,event = None):
        # after button is clicked


        if len(self.tour) == len(self.xy):
            x, y = self.xy[self.tour[-1]]
            dx, dy = self.xy[self.tour[0]] - self.xy[self.tour[-1]]
            self.ax.quiver(
                x, y, dx, dy,
                scale_units='xy',
                angles='xy',
                scale=1,
            )
        else:
            self.tour_generation()

        # 更新图形
        self.fig.canvas.draw()


    def tour_generation(self):
        if len(self.tour) < len(self.xy):
            p = self.oracle(self.tour)  # p is probability(WuTao's notes)
            if self.sample:
                # Advertising the Gumbel-Max trick
                g = -np.log(-np.log(np.random.rand(*p.shape)))
                i = np.argmax(np.log(p) + g)
                # i = np.random.multinomial(1, p)
            else:
                # Greedy
                i = np.argmax(p)
                self.tour.append(i)
            if len(self.tour) == 1:
                x, y = self.xy[self.tour[0]]
                self.ax.scatter(x, y, s=100, color='red')

            elif len(self.tour) <= len(self.xy) :
                x,y = self.xy[self.tour[-2]]
                dx,dy = self.xy[self.tour[-1]]-self.xy[self.tour[-2]]
                self.ax.quiver(
                    x, y, dx, dy,
                    scale_units='xy',
                    angles='xy',
                     scale=1,
                )



if __name__ == '__main__':
    # creat random map, the (x,y) coordinates of each city
    xy = np.random.rand(20, 2)

    # choose model, set evaluate mode
    model, _ = load_model('pretrained/tsp_20/')
    model.eval()  # Put in evaluation mode to not track gradients

    drawer = TSP_Drawer(xy,model)
    plt.show()


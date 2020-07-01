import numpy as np
from neuron import h


class ArtificialCell:
    def __init__(self, event_times):
        # Convert event times into nrn vector
        self.nrn_eventvec = h.Vector()
        self.nrn_eventvec.from_python(event_times)

        # load eventvec into VecStim object
        self.nrn_vecstim = h.VecStim()
        self.nrn_vecstim.play(self.nrn_eventvec)

        # create the cell and artificial NetCon
        self.nrn_netcon = h.NetCon(self.nrn_vecstim, None)


def artificial_cell(event_times):
    # Convert event times into nrn vector
    nrn_eventvec = h.Vector()
    nrn_eventvec.from_python(event_times)

    # load eventvec into VecStim object
    nrn_vecstim = h.VecStim()
    nrn_vecstim.play(nrn_eventvec)

    # create the cell and artificial NetCon
    nrn_netcon = h.NetCon(nrn_vecstim, None)
    return nrn_netcon


def demo(is_class=False, is_list=False):
    print('Begin demo')

    # create parallel context
    pc = h.ParallelContext()
    rank = int(pc.id())

    class OuterClass:
        def __init__(self):

            self.feeds = list()
            for gid in range(20):
                pc.set_gid2node(gid, rank)
                spike_times = np.random.rand(20)
                if is_class:
                    feed = ArtificialCell(spike_times)
                    if is_list:
                        self.feeds.append(feed)
                        pc.cell(gid, self.feeds[-1].nrn_netcon)
                    else:
                        pc.cell(gid, feed.nrn_netcon)
                else:
                    nrn_netcon = artificial_cell(spike_times)
                    if is_list:
                        self.feeds.append(nrn_netcon)
                        pc.cell(gid, self.feeds[-1])
                    else:
                        pc.cell(gid, nrn_netcon)

                print(f'gid={gid}, is_class={is_class}, is_list={is_list}')

    OuterClass()
    pc.gid_clear()
    pc.done()
    print('Completed: demo')


# run demos
################################################
demo(is_class=True, is_list=True)

# none of these work
demo(is_class=False, is_list=True)
demo(is_class=True, is_list=False)
demo(is_class=False, is_list=False)

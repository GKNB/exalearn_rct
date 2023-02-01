import os
import tarfile

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import radical.utils as ru
import radical.pilot as rp
import radical.entk as re
import radical.analytics as ra

plt.style.use(ra.get_mplstyle('radical_mpl'))

os.system("radical-stack")

sid = 'rp.session.thetalogin6.twang3.019339.0001'
#sdir = './'
sdir = '/home/twang3/radical.pilot.sandbox/'
#sdir = 'sessions/'

sp = sdir + sid

session = ra.Session(sp, 'radical.pilot')
#pilots  = session.filter(etype='pilot', inplace=False)
#tasks   = session.filter(etype='task' , inplace=False)

from radical import entk
import os
import argparse, sys, math

class MVP(object):

    def __init__(self):
        self.set_argparse()
        self.set_rmq_env()
        self.am = entk.AppManager(hostname = self.rmq_hostname, port = self.rmq_port,
                                  username = self.rmq_username, password = self.rmq_password)

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

    def set_rmq_env(self):
        self.rmq_port       = int(os.environ.get('RMQ_PORT', 5672))
        self.rmq_hostname   =     os.environ.get('RMQ_HOSTNAME', '95.217.193.116')
        self.rmq_username   =     os.environ.get('RMQ_USERNAME', 'litan')
        self.rmq_password   =     os.environ.get('RMQ_PASSWORD', 'yG2g7WkufPajVUAq')

    def set_argparse(self):
        parser = argparse.ArgumentParser(description = "MVP_entk")
        parser.add_argument("--num_step", "-s", help = "number of training/data generation steps(pairs)")
        parser.add_argument("--num_epoch", "-e", help = "number of epochs")
        parser.add_argument("--num_rank",  "-r", help = "number of MPI ranks")
        args = parser.parse_args()
        self.args = args
        if (args.num_epoch is None) or (args.num_rank is None) or (args.num_step is None):
            parser.print_help()
            sys.exit(-1)

    # This is for training stage
    # return a single stage which includes multiple training tasks for a single point
    def run_training_py(self):

        s = entk.Stage()
        for i in range(int(self.args.num_epoch)):
            t = entk.Task()
            t.pre_exec = [
                    "module load conda/2021-09-22",
                    "export OMP_NUM_THREADS=1"
                    ]
            t.executable = 'python'
            t.arguments = [
                    '/home/twang3/myWork/test/mtnetwork-training-single-epoch.py', 
                    '-i{}'.format(i)
                    ]
            t.post_exec = []
            t.cpu_reqs = {
                    'processes': 1,
                    'process_type': None,
                    'threads_per_process': 1,
                    'thread_type': 'OpenMP'
                    }
            s.add_tasks(t)

        return s

    # This is for simulation stage
    # return a single stage which includes a single simulation task
    def run_simulation_py(self):

        s = entk.Stage()
        t = entk.Task()
        t.pre_exec = [
                "module load conda/2021-09-22",
                "export HDF5_USE_FILE_LOCKING=FALSE",
                "export LD_LIBRARY_PATH=/home/twang3/g2full_theta/GSASII/bindist:$LD_LIBRARY_PATH"
                ]
        t.executable = 'python'
        t.arguments = [
                '/home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py', 
                '/home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/configs/config_1521500_monoclinic.txt'
                ]
        t.post_exec = []
        t.cpu_reqs = {
                'processes': self.args.num_rank,
                'process_type': 'MPI',
                'threads_per_process': 1,
                'thread_type': 'OpenMP'
                }
        s.add_tasks(t)

        return s

    def run_workflow_trivial(self):

        p = entk.Pipeline()
        for i in range(int(self.args.num_step)):
            s1 = self.run_simulation_py()
            p.add_stages(s1)
            s2 = self.run_training_py()
            p.add_stages(s2)

        self.am.workflow = [p]
        self.am.run()


if __name__ == "__main__":

    mvp = MVP()
    n_nodes = math.ceil(float(int(mvp.args.num_rank)/64))
    mvp.set_resource(res_desc = {
        'resource': 'anl.theta',
        'queue'   : 'debug-flat-quad',
        'walltime': 60, #MIN
        'cpus'    : 64 * n_nodes,
        'gpus'    : 0 * n_nodes,
        'project' : 'CSC249ADCD08'
        })
    mvp.run_workflow_trivial()

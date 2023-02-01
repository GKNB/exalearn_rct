from radical.entk import Pipeline, Stage, Task, AppManager
import os
import argparse, sys, math

class MVP(object):

    def __init__(self):
        self.set_argparse()
        self.set_rmq_env()
        self.am = AppManager(hostname = self.rmq_hostname, port = self.rmq_port,
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

    # This is for simulation, return a single task rather than a stage!
    def simulation_task():

        t = Task()
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

        return t

    # This is for training, return a single stage including multiple tasks!
    # We can then include more tasks
    def training_stage(self):

        s = Stage()
        for i in range(int(self.args.num_epoch)):
            t = Task()
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

    def generate_pipeline():

        # Create a Pipeline object
        p = Pipeline()

        s1 = Stage()
        s1.name = 'Stage.pre'

        t1 = self.simulation_task()
        s1.add_tasks(t1)
        p.add_stages(s1)

        for s_cnt in range(int(self.args.num_step)):

            # Create a Stage object
            s = self.training_stage()
            s.name = 'Stage.%s' % s_cnt

            # Add the simulation task to the training stage
            t = self.simulation_task()
            s.add_tasks(t)

            p.add_stages(s)

        s2 = self.training_stage()
        s2.name = 'Stage.post'

        # Add Stage to the Pipeline
        p.add_stages(s2)

        return p

    def run_workflow_sol_2(self):

        p = self.generate_pipeline()
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
    mvp.run_workflow_sol_2()

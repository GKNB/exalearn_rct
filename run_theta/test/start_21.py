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

    # This is to combine simulation and training into a single pipeline
    def generate_pipeline(index):

        p = Pipeline()
        p.name = 'Pipeline.%s' % index
    
        if index != 0:
            s0 = Stage()
            s0.name = 'Pipeline.%s.Stage.0.monitor' % index
    
            t0 = Task()
            t0.name = 'Pipeline.%s.Task.0.monitor' % index
            t0.executable = '/bin/bash'
            t0.arguments = ['/home/twang3/myWork/test/look_for_done.sh', '/home/twang3/myWork/test/temp_touch', '%s' % (index-1)]
    
            s0.add_tasks(t0)
            p.add_stages(s0)

        # This is simulation stage
        s1 = Stage()
        s1.name = 'Pipeline.%s.simulation' % index
 
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
  
        # Generate a file which says this work is done
        t.post_exec = ['touch /home/twang3/myWork/test/temp_touch/Done.task.%s' % index]
    
        s1.add_tasks(t)
        p.add_stages(s1)
    
        # This is for training stage
        s2 = Stage()
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
            s2.add_tasks(t)

        p.add_stages(s2)
        return p

    def run_workflow_sol_2(self):

        pipelines = []
        for i in range(int(self.args.num_step)):
            p = self.generate_pipeline(i)
            pipelines.append(p)
        self.am.workflow = set(pipelines)
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

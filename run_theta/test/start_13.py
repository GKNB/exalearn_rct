from radical import entk
import os
import argparse, sys, math

class MVP(object):

    def __init__(self):
        self.set_argparse()
        self._set_rmq()
        self.am = entk.AppManager(hostname=self.rmq_hostname, port=self.rmq_port, username=self.rmq_username, password=self.rmq_password)
        self.p = entk.Pipeline()
        self.s = entk.Stage()

    def _set_rmq(self):
        self.rmq_port = int(os.environ.get('RMQ_PORT', 5672))
        self.rmq_hostname = os.environ.get('RMQ_HOSTNAME', '95.217.193.116')
        self.rmq_username = os.environ.get('RMQ_USERNAME', 'litan')
        self.rmq_password = os.environ.get('RMQ_PASSWORD', 'yG2g7WkufPajVUAq')

    def set_resource(self, res_desc):
        res_desc["schema"] = "local"
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="MVP_EnTK")
        parser.add_argument("--num_rank", "-n", help="number of MPI ranks")
        args = parser.parse_args()
        self.args = args
        if args.num_rank is None:
            parser.print_help()
            sys.exit(-1)

    def run_mpi_sweep_hdf5_py(self, num_rank):

        ntask = 1
        nproc = int(int(num_rank)/ntask)
        for i in range(ntask):#range(1, int(num_sim) + 1):
            t = entk.Task()
            t.pre_exec = [
                #"source /home/litan/miniconda3/etc/profile.d/conda.sh",
                "module load conda/2021-09-22",
                #"conda activate mvp2",
                #"echo \"RP_RANKS=$RP_RANKS\"; echo \"MPI_RANK=$MPI_RANK\"; echo \"PMIX_RANK=$PMIX_RANK\"",
                "export HDF5_USE_FILE_LOCKING=FALSE",
                "export LD_LIBRARY_PATH=/home/twang3/g2full_theta/GSASII/bindist:$LD_LIBRARY_PATH",
                "echo $LD_LIBRARY_PATH"
                #"export OMP_NUM_THREADS=1"
                ]
            t.executable = 'python'
            t.arguments = ['/home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py', 
                           '/home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/configs/config_1521500_monoclinic.txt']#'-n{}'.format(num_sim)
            #t.post_exec = ["export TASK_ID={}".format(t.uid),"echo $TASK_ID | cut -d \".\" -f 2"]
            t.post_exec = []
            t.cpu_reqs = {
                'processes': nproc,
                'process_type': 'MPI',#None
                'threads_per_process': 1,
                'thread_type': 'OpenMP'
            }
            '''t.gpu_reqs = {
                'processes': 0,
                'process_type': None,
                'threads_per_process': 1,
                'thread_type': 'CUDA'
            }'''
            self.s.add_tasks(t)
        self.p.add_stages(self.s)

    def run(self):
        self.am.workflow = [self.p]
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
    mvp.run_mpi_sweep_hdf5_py(num_rank=mvp.args.num_rank)
    mvp.run()

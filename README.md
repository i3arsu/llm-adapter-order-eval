# llm-adapter-order-eval

## HPT commands

start build
qsub script.pbs


script.pbs

example



check build
qstat -u username


track terminal output (once script starts running and file is generated)
tail -f fileoutputname (i.e. filename.o3427302)


stop script | remove script from queue
qdel jobid (just first part)

force stop
qdel -W force jobid
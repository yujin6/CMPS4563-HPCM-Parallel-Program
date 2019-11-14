/*******************************************************************************
* ICA2
* Name: Joshua Allen, Yujin
* Parallel Programming Date: 10/30/2019
********************************************************************************
* Description
*******************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    int comm_sz, rank;

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int p= 191;

    if(rank == 0)
    {
        MPI_Send(&p, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&p, 1, MPI_INT, comm_sz-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Message recieved is: %d\n", p);
    }
    else if(rank != comm_sz-1)
    {
        MPI_Recv(&p, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        p+= 111*rank;
        MPI_Send(&p, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Recv(&p, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        p+=111*rank;
        MPI_Send(&p, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }


    return 0;
}

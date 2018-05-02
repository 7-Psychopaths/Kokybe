/*
 * File:   main.cpp
 * Author: mindaugas
 *
 * Created on August 28, 2013, 9:57 PM
 */

#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include "ARFF.h"
#include "Statistics.h"
#include "ShufleEnum.h"
#include "DistanceMetrics.h"
#include "ObjectMatrix.h"
#include "MDS.h"
#include "HPCMethod.h"
#include "DimReductionMethod.h"
#include "SMACOF.h"
#include "SAMANN.h"
#include "SMACOFZEIDEL.h"
#include "SDS.h"
#include "SOM.h"
#include "SOMMDS.h"
#include "DMA.h"
#include <mpi.h>
#include "Projection.h"
#include "PCA.h"
#include "AdditionalMethods.h"
#include "KMEANS.h"
#include "MLP.h"
#include "DECTREE.h"
#include "cmdLineParser/CommandLineParser.h"
#include <time.h>

#include <sstream>
#include <algorithm>

int AdditionalMethods::PID;    //  seed for random numbers generator
std::string AdditionalMethods::inputDataFile = ""; // input data file that will be passed to the HPC method constructor

                                                                                // Y atvaizdavimas ekrane (testavimui)
double strToDouble(std::string);                                                // command line parameter to double type
void FindAlgorithm(std::string &tmp, CommandLineParser &cmdLine, int pid, int numOfProcs, std::string &resultFile, std::string &statFile);
int strToInt(std::string cmdParam);

template <typename T>                                             // command line parameter to int type
void paralelCompute(int pid, int numOfProcs, T *mthd, std::string resultFile, std::string statFile); // call functiopn

int main(int argc, char** argv)
{

    std::string inputFile = "", resultFile="", statFile="";     // pradiniu duomenu, rezultatų ir paklaidų failai
    std::string tmp ="";
    const char* send;                                        //temp parameter for call method selection
    int tmpPathSize;

    int numOfProcs, pid; // (numOfProcs) procesu kiekis

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    if (pid == 0)
        time(&AdditionalMethods::startTime);

    AdditionalMethods::PID = pid;

    CommandLineParser cmdLine(argc,argv,true);

    //extract command line arguments

    inputFile = cmdLine.get_arg("-i");
    resultFile = cmdLine.get_arg("-o");
    statFile = cmdLine.get_arg("-s");

    if (!inputFile.empty() || !resultFile.empty() || !statFile.empty())
    {
        AdditionalMethods::inputDataFile.assign(inputFile);

        //generate file name for X distance matrix
            AdditionalMethods::tempFileSavePath = strdup(AdditionalMethods::generateFileName().c_str());

            if (AdditionalMethods::tempFileSavePath == "")
            {
                MPI_Finalize();
                std::cout <<"Unable to generate file name for X distance matrix";
                return 0;
            }


        tmp = cmdLine.get_arg("-al");
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::toupper);

        Statistics::initSeed();
		FindAlgorithm(tmp, cmdLine, pid, numOfProcs, resultFile, statFile);
    }
    else
    {
        std::cout << "Input/output file parameter(s) not found" << std::endl;
    }


    MPI_Finalize();
    return 0;
}

void FindAlgorithm(std::string &tmp, CommandLineParser &cmdLine, int pid, int numOfProcs, std::string &resultFile, std::string &statFile)
{
	switch (tmp) // <- ThisMM
	{
	case "PCA":
		tmp = cmdLine.get_arg("-projType");
		std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::toupper);

		if (tmp == "0" || tmp == "FALSE")
		{
			PCA_ *method = new PCA_(strToInt(cmdLine.get_arg("-d")), false);
			paralelCompute(pid, numOfProcs, method, resultFile, statFile);
		}
		else
		{
			PCA_ *method = new PCA_(strToDouble(cmdLine.get_arg("-d")), true);

			paralelCompute(pid, numOfProcs, method, resultFile, statFile);
		}
		break;
	case "DMA":
		DMA * method = new DMA(strToDouble(cmdLine.get_arg("-eps")), strToInt(cmdLine.get_arg("-maxIter")), strToInt(cmdLine.get_arg("-d")),
        strToInt(cmdLine.get_arg("-neighbour")));
		paralelCompute(pid, numOfProcs, method, resultFile, statFile);
		break;
	case "RELATIVEMDS":
		int pEnum = strToInt(cmdLine.get_arg("-selStrategy"));
		SDS *method;

		if (pEnum == 1)
			method = new SDS(strToDouble(cmdLine.get_arg("-eps")), strToInt(cmdLine.get_arg("-maxIter")), strToInt(cmdLine.get_arg("-d")),
			        ProjectionEnum::RAND, strToInt(cmdLine.get_arg("-noOfBaseVectors")), EUCLIDEAN); //may be manhatan or chebyshew
		else if (pEnum == 2)
			method = new SDS(strToDouble(cmdLine.get_arg("-eps")), strToInt(cmdLine.get_arg("-maxIter")), strToInt(cmdLine.get_arg("-d")),
			        ProjectionEnum::PCA, strToInt(cmdLine.get_arg("-noOfBaseVectors")), EUCLIDEAN); //may be manhatan or chebyshew
		else
			method = new SDS(strToDouble(cmdLine.get_arg("-eps")), strToInt(cmdLine.get_arg("-maxIter")), strToInt(cmdLine.get_arg("-d")),
			        ProjectionEnum::DISPERSION, strToInt(cmdLine.get_arg("-noOfBaseVectors")), EUCLIDEAN); //may be manhatan or chebyshew

																																																								  //std::cout << strToDouble(cmdLine.get_arg("-eps")) <<" "<< strToInt(cmdLine.get_arg("-maxIter")) <<" "<< strToInt(cmdLine.get_arg("-d")) <<" "<< strToInt(cmdLine.get_arg("-selStrategy")) <<" "<< strToInt(cmdLine.get_arg("-noOfBaseVectors")) <<" "<< EUCLIDEAN;
		paralelCompute(pid, numOfProcs, method, resultFile, statFile);
		break;
	case "SMACOFMDS":
		tmp = cmdLine.get_arg("-zeidel");
		std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::toupper);
		if (tmp == "0" || tmp == "FALSE")
		{
			double eps = strToDouble(cmdLine.get_arg("-eps"));
			int maxIter = strToInt(cmdLine.get_arg("-maxIter"));
			int d = strToInt(cmdLine.get_arg("-d"));

			SMACOF *method = new SMACOF(eps, maxIter, d);
			paralelCompute(pid, numOfProcs, method, resultFile, statFile);
		}
		else
		{
			SMACOFZEIDEL *method = new SMACOFZEIDEL(strToDouble(cmdLine.get_arg("-eps")), strToInt(cmdLine.get_arg("-maxIter")),
			        strToInt(cmdLine.get_arg("-d")), RANDOM); // sorting method may be other
																																										   //std::cout << strToDouble(cmdLine.get_arg("-eps")) << strToInt(cmdLine.get_arg("-maxIter")) << strToInt(cmdLine.get_arg("-d")) << BUBLESORTDSC << (cmdLine.get_arg("-zeidel"));
			paralelCompute(pid, numOfProcs, method, resultFile, statFile);
		}
		break;
	case "SAMANN":
		SAMANN * method = new SAMANN(strToInt(cmdLine.get_arg("-mTrain")), strToInt(cmdLine.get_arg("-nNeurons")),
		        strToDouble(cmdLine.get_arg("-eta")), strToInt(cmdLine.get_arg("-maxIter")));
		paralelCompute(pid, numOfProcs, method, resultFile, statFile);
		break;
	case "SOMMDS":
		SOMMDS * method = new SOMMDS(strToDouble(cmdLine.get_arg("-eps")), strToInt(cmdLine.get_arg("-maxIter")),
		        strToInt(cmdLine.get_arg("-mdsProjection")), strToInt(cmdLine.get_arg("-rows")), strToInt(cmdLine.get_arg("-columns")),
                                     strToInt(cmdLine.get_arg("-eHat")));
		//method.
		paralelCompute(pid, numOfProcs, method, resultFile, statFile);
		break;
	case "SOM":
		SOM * method = new SOM(strToInt(cmdLine.get_arg("-rows")), strToInt(cmdLine.get_arg("-columns")), strToInt(cmdLine.get_arg("-eHat")));
		paralelCompute(pid, numOfProcs, method, resultFile, statFile);
		break;
	case "KMEANS":
		KMEANS * method = new KMEANS(strToInt(cmdLine.get_arg("-noOfClust")), strToInt(cmdLine.get_arg("-maxIter")));
		paralelCompute(pid, numOfProcs, method, resultFile, statFile);
		break;
	case "MLP":
		tmp = cmdLine.get_arg("-kFoldVal");
		std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::toupper);
		bool kFoldValidation = true;
		if (tmp == "0" || tmp == "FALSE")
		{
			kFoldValidation = false;
		}
		MLP *method = new MLP(strToInt(cmdLine.get_arg("-h1pNo")), strToInt(cmdLine.get_arg("-h2pNo")), strToDouble(cmdLine.get_arg("-qty")),
		        strToInt(cmdLine.get_arg("-maxIter")), kFoldValidation);
		paralelCompute(pid, numOfProcs, method, resultFile, statFile);
	case "DECTREE":
		DECTREE * method = new DECTREE(strToDouble(cmdLine.get_arg("-dL")), strToDouble(cmdLine.get_arg("-dT")),
		        strToDouble(cmdLine.get_arg("-r")), strToInt(cmdLine.get_arg("-nTree")));
		paralelCompute(pid, numOfProcs, method, resultFile, statFile);
	default:
		std::cout << "Unknown algorithm call" << std::endl;
	}
}

template <typename T>
void paralelCompute(int pid, int numOfProcs, T *mthd, std::string resultFile, std::string statFile)
{
    time_t t_end;                      // skaiciavimu pradzia ir pabaiga
    ObjectMatrix Y;                             // projekcijos matrica
    double *stressErrors;                       // surinktu is procesu paklaidu aibe (testavimui)
    double receivedStress, min_stress = 0.0;    // gaunama ir maziausia paklaidos
    double **receiveArray, **sendArray;         // gaunama ir siunciama Y matricos

    MPI_Status status;

    int send, min_rank = 0;

    if (pid == 0)
    {
        if (numOfProcs == 1)
        {
            Y = mthd->getProjection();
            Y.saveDataMatrix(resultFile.c_str());
            ARFF::writeStatData(statFile, mthd->getStress(), difftime(time(&t_end), AdditionalMethods::startTime));
        }
        else
        {
            stressErrors = new double[numOfProcs];     // surinktu paklaidu masyvas (testavimui)
            Y = mthd->getProjection();
            int n ;//= Y.getObjectCount();
            int m ;//= Y.getObjectAt(0).getFeatureCount();

            stressErrors[0] = mthd->getStress()+1;
            for (int i = 1; i < numOfProcs; i++)
            {
                MPI_Recv(&receivedStress, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);  // priimama paklaida is kiekvieno proceso
                stressErrors[i] = receivedStress;
            }

            min_stress = stressErrors[0];
            for (int i = 1; i < numOfProcs; i++)
               {
                   if (stressErrors[i] < min_stress)
                {
                    min_stress = stressErrors[i];
                    min_rank = i;

                }
               }
            if (min_rank == 0)  // jei maziausia paklaida tevinio proceso siunciamas pranesimas likusiems, kad savo Y nesiustu
            {
                send = 0;

                for (int i = 1; i < numOfProcs; i++)
                    MPI_Send(&send, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                Y.saveDataMatrix(resultFile.c_str());
                ARFF::writeStatData(statFile, stressErrors[min_rank], difftime(time(&t_end), AdditionalMethods::startTime));
            }
            else
            {
                for (int i = 1; i < numOfProcs; i++)
                    if (i == min_rank)
                    {
                        send = 1;
                        MPI_Send(&send, 1, MPI_INT, i, 0, MPI_COMM_WORLD);  // siunciamas pranesimas, kad atsiustu Y
                    }
                    else
                    {
                        send = 0;
                        MPI_Send(&send, 1, MPI_INT, i, 0, MPI_COMM_WORLD);  // siunciamas pranesimas, kad nesiustu Y
                    }


                    MPI_Recv(&n, 1, MPI_INT, min_rank, min_rank, MPI_COMM_WORLD, &status);
                    MPI_Recv(&m, 1, MPI_INT, min_rank, min_rank, MPI_COMM_WORLD, &status);
                receiveArray = AdditionalMethods::Array2D(n, m);
                MPI_Recv(&(receiveArray[0][0]), n * m, MPI_DOUBLE, min_rank, min_rank, MPI_COMM_WORLD, &status);   // priimama Y data dali
                Y = AdditionalMethods::DoubleToObjectMatrix(receiveArray, n, m);
                //update the Y matrix data that was received from the another process
                for (int i = 0; i < n; i++)
                        for (int j = 0; j < m; j++)
                            Y.updateDataObject(i,j,receiveArray[i][j]);
                int classQty = 0;
                MPI_Recv(&classQty, 1, MPI_INT, min_rank, min_rank, MPI_COMM_WORLD, &status); // receive the different class qty


                if (classQty !=0 )
                {
                    std::vector<std::string> receivedAtrrClasses;
                    int clNameLenght;
                    for (int cl = 0; cl < classQty; cl++)
                    {
                        MPI_Recv(&clNameLenght, 1, MPI_INT, min_rank, min_rank, MPI_COMM_WORLD, &status); // must send +1
                        char buffer[clNameLenght];
                        MPI_Recv(buffer, clNameLenght, MPI_CHAR, min_rank, min_rank, MPI_COMM_WORLD, &status);
                        receivedAtrrClasses.push_back(std::string(buffer));
                    }
                    Y.setPrintClass(receivedAtrrClasses);

                    int classLabelBuffer[n];
                    MPI_Recv(&classLabelBuffer, n, MPI_INT, min_rank, min_rank, MPI_COMM_WORLD, &status);
                    for (int i = 0; i < n; i++)
                        {
                            Y.updateDataObjectClass(i,classLabelBuffer[i]);
                        }
                }
                Y.saveDataMatrix(resultFile.c_str());
                ARFF::writeStatData(statFile, stressErrors[min_rank], difftime(time(&t_end), AdditionalMethods::startTime));
            }
        }
    }
    else
    {
        Y = mthd->getProjection();

        double stress = mthd->getStress();
        MPI_Send(&stress, 1, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD);  // siunciama paklaida teviniam procesui
        MPI_Recv(&send, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);  // priimamas pranesimas ar siusti Y


        if (send == 1)          // siusti, jei send = 1, nesiusti, jei send = 0
        {
            int n = Y.getObjectCount();
            int m = Y.getObjectAt(0).getFeatureCount();
            //siisiunciam busimos matricos dydi //SOM atveju nugal4toj7 kiekis gali ir nesutapti su tevinio porceso turimu
            MPI_Send(&n, 1, MPI_INT, 0, pid, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, 0, pid, MPI_COMM_WORLD);

            sendArray = AdditionalMethods::Array2D(n, m);
            sendArray = AdditionalMethods::ObjectMatrixToDouble(Y);
            MPI_Send(&(sendArray[0][0]), m * n, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD);  // siunciama Y

            int classQty = Y.getClassCount();
            MPI_Send(&classQty, 1, MPI_INT, 0, pid, MPI_COMM_WORLD); //siunciam kiek  yra skirtingu klasiu atributu sekcijoje
            if (classQty != 0 )
            {
                std::vector<std::string> possClasses = Y.getStringClassAttributes();
                for (int i = 0 ; i < classQty; i ++)
                {
                    int clsSize = possClasses.at(i).size()+1;
                    MPI_Send(&clsSize, 1, MPI_INT, 0, pid, MPI_COMM_WORLD); //siunciam charo dydi
                    MPI_Send((void*)possClasses.at(i).c_str(), clsSize, MPI_CHAR, 0, pid, MPI_COMM_WORLD); //siunciam patį stringą
                }
                //kadangi yra ir pačios klases siunciam ir objekto klasių masyvą
                int classLabelBuffer[n];
                for (int i = 0; i < n; i++)
                    classLabelBuffer[i] = Y.getObjectAt(i).getClassLabel();

                MPI_Send(&classLabelBuffer, n, MPI_INT, 0, pid, MPI_COMM_WORLD); //send buffer
            }

        }

    }
}
/*
* Method that converts string command line parameter to doublel
*/
double strToDouble(std::string cmdParam)
{
    const char *str = cmdParam.c_str();
    char *err;
    double x = strtod(str, &err);
    if (*err == 0 && cmdParam !="")
    {
        return atof(cmdParam.c_str());
    }
}

int strToInt(std::string cmdParam)
{
    const char *str = cmdParam.c_str();
    char *err;
    double x = strtod(str, &err);
    if (*err == 0 && cmdParam !="")
    {
        return atoi(cmdParam.c_str());
    }
}

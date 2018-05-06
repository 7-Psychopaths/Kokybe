#include "MLP.h"

MLP::MLP()
{
    //ctor
}
MLP::MLP(int h1pNo, int h2pNo, double qtty, int maxIter, bool validationMethod)
{
    //no of neurons in hidden layer
    HiddenNeuronsNo1 = h1pNo;
    HiddenNeuronsNo2 = h2pNo;
    this->maxIter = maxIter;
    this->kFoldValidation = validationMethod;

    alglib::mlpcreatetrainercls(X.getObjectAt(0).getFeatureCount(), X.getClassCount(), trn);
    //qtt of input features, number of classes to be produced

    double wstep = 0.000;
    mlpsetdecay(trn, 0.001); // by default we set moderate weight decay
    mlpsetcond(trn, wstep, this->maxIter);     // * we choose iterations limit as stopping condition
    //(another condition - step size - is zero, which means than this condition is not active)

    if ((HiddenNeuronsNo1 > 0) && (HiddenNeuronsNo2 > 0))
        alglib::mlpcreatec2(X.getObjectAt(0).getFeatureCount(), HiddenNeuronsNo1, HiddenNeuronsNo2, X.getClassCount(), network);
    //create nn network with noofinput features, 2 hidden layers, noofclasses (and sore to network variable)
    if ((HiddenNeuronsNo1 > 0) && (HiddenNeuronsNo2 == 0))
        alglib::mlpcreatec1(X.getObjectAt(0).getFeatureCount(), HiddenNeuronsNo1, X.getClassCount(), network);
    //create nn network with no of input features, 1 hidden layer, noofclasses (and sore to network variable)
    if ((HiddenNeuronsNo1 == 0) && (HiddenNeuronsNo2 == 0))
        alglib::mlpcreatec0(X.getObjectAt(0).getFeatureCount(), X.getClassCount(), network);
    //create nn network with no of input features, 0 hidden layer, noofclasses (and sore to network variable)
///HiddenNeuronsNo2 must be non zero

    if (this->kFoldValidation == true) //do kfold validation
    {
        ClusterizationMethods::initializeData();

        alglib::mlpsetdataset(trn, ClusterizationMethods::learnSet, ClusterizationMethods::learnObjQtty); //attach learning data to data set

        alglib::mlpkfoldcv(trn, network, 1, int(qtty), rep);
    }
    else
    {
        ClusterizationMethods::initializeData(qtty, 100 - qtty);

        alglib::mlpsetdataset(trn, ClusterizationMethods::learnSet, ClusterizationMethods::learnObjQtty); //attach learning data to data set

        alglib::mlptrainnetwork(trn, network, 1, rep);
        // train network NRestarts=1, network is trained from random initial state.
        //With NRestarts=0, network is trained without randomization (original state is used as initial point).
        alglib::integer_1d_array Subset;
        Subset.setlength(10);
        alglib::mlpallerrorssubset(network, testSet, testObjQtty, Subset, -1, repp);
    }
    //ctor
    // now get network error // do not calculate cross-validation since it validates the topology of the network
}

MLP::~MLP()
{
    //dtor
}

ObjectMatrix MLP::getProjection()
{
    int featureCount = X.getObjectAt(0).getFeatureCount();
    int objCount = X.getObjectCount();

    initializeYMatrix(objCount, featureCount + X.getClassCount());

    alglib::real_1d_array tmpYObj;
    alglib::real_1d_array tmpXObj;

    tmpYObj.setlength(featureCount);
    tmpXObj.setlength(X.getClassCount());

    DataObject tempObj;
    for (int i = 0; i < objCount; i++)
    {
        tempObj = X.getObjectAt(i);
        for (int ft = 0; ft < featureCount; ft++)
        {
            double feature = tempObj.getFeatureAt(ft);
            tmpYObj(ft) = feature;
            Y.updateDataObject(i, ft, feature);
        }

        alglib::mlpprocess(network, tmpYObj, tmpXObj);

        double max_prob = tmpXObj(0);
        int indx = 0;

        for (int j = 0; j < X.getClassCount(); j++)
        {
            Y.updateDataObject(i, j + featureCount, tmpXObj(j));
            if (max_prob < tmpXObj(j))
            {
                max_prob = tmpXObj(j);
                indx = j;
            }
        }

        if (tempObj.getClassLabel() != -1)
            Y.updateDataObjectClass(i, tempObj.getClassLabel());
        else
            Y.updateDataObjectClass(i, indx);
    }

    std::vector <std::string > probabilities; probabilities.reserve(0);

    for (int i = 0; i < X.getClassCount(); i++)
        probabilities.push_back("probClass" + X.getStringClassAttributes().at(i));

    Y.addAtributes(probabilities);

    Y.setPrintClass(X.getStringClassAttributes());

    return Y;
}

double MLP::getStress()
{
    if (this->kFoldValidation) return rep.avgrelerror;
    else return repp.avgrelerror;
//}
    /*
    * Rep.RelCLSError - fraction of misclassified cases.
    * Rep.AvgCE - acerage cross-entropy
    * Rep.RMSError - root-mean-square error
    * Rep.AvgError - average error
    * Rep.AvgRelError - average relative error
    */
    return rep.rmserror;
}

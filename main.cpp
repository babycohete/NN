/*

                Back-propagation Deep (Adjustable) Network
                            (OOP Approach)

by Sarah Aguasvivas-Manzano

June 2016

Assumptions:

-   This program assumes that the neural network is a sigmoid neural network.

-   Example file has data for the XOR problem, meaning 3 neurons in the
input layer and 1 neuron in the output layer.

-	Input and desired output come from a data file.
 
- Using Haykin's book. For notation purposes:
 
 What Haykin refers to as v_j(n), I called it list[number_layer].out[j]
 What Haykin refers to as y_j(n), I called it list[number_layer].in[j]

*/

#include <iostream>
#include <iomanip>
#include <array>
#include <sstream>
#include <cmath>
#include <string>
#include <ctime>
#include <ctype.h>
#include <fstream>
#include <vector>
#include <cstdlib>


using namespace std;

int numNN = 0;
const double a = 0.9;       //optimum momentum 1.7159
const double eta = 1e-3;        //learning rate 1e-3 for MNIST
const double Tol = 5e-2;       //Tolerance  5e-2 for MNIST

double Sigmoid(double);
double SigmoidPrime(double);
void readData(double, int &, int &);
int inputHidden();
void inputStuff(int, int *);

class NeuralNet {
	//This contains general information about the Neural Network.

public:
	int  numH, layers, *numNeurons;
	void initNetwork(int);
	double error;
	double *des = NULL, *inp=NULL;

	//constructor
	NeuralNet(){
		numNN++;
		numNeurons = new int [layers + 2];
	}
};

class Layer : public NeuralNet {

	/*The operational part of my code takes place in the layer class. 
	There is no neuron class because it is more practical to store information
	as an array of neurons than neurons per se.*/

public:

	void operator+(Layer a);
	void operator=(Layer a);
	int n, m;
	double *in=NULL, *out=NULL;
	double **w =NULL, *b = NULL, **w_temp;
	void initWeights(int, int);
    double *delta;

	Layer(int n1, int m1){

		layers++;
		//cout << "Layer created... " << endl;
		n = n1;
		m = m1;
			in = new double[m];
			out = new double[m];
            delta= new double [m];
        
			//Initializing input/output:
		for (int i = 0; i < m; i++) {
			in[i] = 0.0;
			out[i] = 0.0;
		}
        if (n!= 0 && m!=0) {
            
            w= new double *[n];
            for(int i=0; i<n; i++) w[i]= new double [m];
            b = new double[m];
            
            w_temp= new double *[n];
            for(int i=0; i<n; i++) w_temp[i]= new double [m];
            
            initWeights(n, m);
        }
	}

	~Layer(){
	}

};

void Layer::operator+(Layer b){
	
	/* Weighted sum to pass information between current layer and layer b.
	This will be used later in the FwdPass subroutine. */

	double *v = new double[b.m];

	for (int i = 0; i<b.m; i++) v[i] = 0.0;
    

    for (int j = 0; j<b.m; j++){
        for (int i = 0; i<b.n; i++){
            v[j] += b.w[i][j] * this->out[i];
        }
        b.in[j] = v[j] + b.b[j];
        b.out[j] = Sigmoid(b.in[j]);
        
    }
}

void Layer::operator=(Layer b) {
    
	/*This was created in case I needed to copy an element. */

	b.n = this->n;
	b.m = this->m;
	b.numNeurons = this->numNeurons;

	for (int i = 0; i < this->n; i++) {
		for (int j = 0; j < this->m; j++) {
			b.w[i][j] = this->w[i][j];
		}
		b.b[i] = this->b[i];
		b.in[i] = this->in[i];
	}

	for (int i = 0; i > this->m; i++) b.out[i] = this->out[i];

}

void NeuralNet::initNetwork(int H) {
	numH = H;          //number of hidden layers
	layers = numH + 2; //adding input and output layers
}

void Layer::initWeights(int n, int m){

    
	double n0 = n;
	double n1 = m;
	double u = 1.0 / n0, gamma, v, sum = 0.0;

	gamma = 0.7*pow(n1, u);
    
	for (int i = 0; i<n; i++){
		for (int j = 0; j<m; j++){
         
			v = 1.0*((double)rand()) / ((double)RAND_MAX) - 0.5;
			w[i][j] = v;
			sum += pow(w[i][j], 2);
			w[i][j] = 1e-0*gamma*w[i][j] / (pow(sum, 0.5));
			b[j] = 2.0*w[i][j]*((double)rand()/((double)RAND_MAX))- w[i][j];
		}
	}
}

double Sigmoid(double x){
	return 1.0 /(1.0 + exp(-a*x));
}

double SigmoidPrime(double x){
	return Sigmoid(x)*(1.0-Sigmoid(x));
}

void readData(double **arrayA, int & rowA, int & colA){

	string lineA;
	double x;
	bool fai = 1 ;

	string datafile;
	ifstream fileIN;
	do {
		//Entering the data file name:
		cout << "\nThe data file that I used for the code is 'data.txt'" << endl;
		cout << "Enter data file name i.e. filename.ext: " << endl;
        
		cin >> datafile;

		fileIN.open(datafile);

		if (fileIN.fail()){
			cout << "**********************************" << endl;
			cerr << "\nError at opening file :( ";
			fai = 1;
		}
		else fai = 0;

	} while (fai==1);

	//Readig data:
    
    rowA= 0;
    colA= 0;
    
	while (fileIN.good()){
		while (getline(fileIN, lineA)){
			istringstream streamA(lineA);
			colA = 0;
			while (streamA >> x){
				arrayA[rowA][colA] = x;
				colA++;
			}
			rowA++;
		}
	}
}

int inputHidden(){
	int an;
	bool bFail = true;
	do{
		cout << "Enter amount of hidden layers: ";
		cin >> an;
		bFail = cin.fail();
		cin.clear();
		cin.ignore(numeric_limits<streamsize>::max(), '\n');

	} while (an <= 0 || bFail == true);
	return an;
}

void inputStuff(int layers, int *array){

	bool bFail = true;

	for (int i = 0; i<layers; i++){
		if (i == 0){

			bFail = true;
			do{
				cout << "Enter # of neurons in input layer: ";
				cin >> array[0];

				bFail = cin.fail();
				cin.clear();
				cin.ignore(numeric_limits<streamsize>::max(), '\n');

			} while (array[0] <= 0 || bFail == true);
		}
		else if (i != layers - 1){

			bFail = true;

			do{
				cout << "Enter # of neurons in hidden layer #" << i << ": ";
				cin >> array[i];

				bFail = cin.fail();
				cin.clear();
				cin.ignore(numeric_limits<streamsize>::max(), '\n');

			} while (array[i] <= 0 || bFail == true);
		}
		else {
			bFail = true;
			do{
				cout << "Enter amount of neurons in output layer: ";
				cin >> array[layers - 1];

				bFail = cin.fail();
				cin.clear();
				cin.ignore(numeric_limits<streamsize>::max(), '\n');

			} while (array[layers - 1] <= 0 || bFail == true);
		}
	}
}

void initLayers(NeuralNet & NN, int *x, vector < Layer > &list){
	
    /*This will store the layers with their corresponding numbers of
	neurons inside the vector called 'list'. It is important to know that in
	this context, m is the amounts of neurons before the layer and m is the amount
	of neurons in the layer. This convention will ease the computations. 
	*/

	NN.numNeurons[0] = 0;
		for (int i = 0; i < NN.layers+1; i++){
			NN.numNeurons[i+1] = x[i];
		}
		NN.numNeurons[NN.layers+1] = 0;

		for (int ii = 0; ii < NN.layers; ii++){
			Layer lay(NN.numNeurons[ii], NN.numNeurons[ii+1]);
			list.push_back(lay);
		}
}

int getLargest(vector <Layer > &list){
    
   /* Getting largest amount of neurons (m) through all the network for
    further use. */
    
    int larg1 = 0;
    
    for (int i=0; i< list.size(); i++){
        if(list[i].m > larg1) larg1 = list[i].m;
    }
    
    return larg1;
}

void FwdPass(vector < Layer > &list) {

    /* This part of the code performs a weighted sum of the values in
     the layers of the NN in a forward way */
    
    for (int i = 0; i < ((int)list.size()); i++) {
        list[i] + list[i + 1];
    }

}

void localGrad(vector <Layer> &list, int larg, NeuralNet & NN, double *err, int randRow, int rowA){
   
    double sum_temp = 0.0;
    double *temp= new double[larg];
   
    //Storing temporal array:
    for (int k=1; k< (int)list.size(); k++){
        for (int i=0; i< list[k].n; i++){
            for (int j=0; j< list[k].m; j++){
                list[k].w_temp[i][j]= list[k].w[i][j];
            }
        }
    }

    //local gradient for output layer:
    for (int i = 0; i < list[list.size()-1].m; i++){
        sum_temp += pow((NN.des[i] - list[list.size() - 1].out[i]), 2);
        list[list.size()-1].delta[i] = (NN.des[i] - list[list.size() - 1].out[i]) * SigmoidPrime(list[list.size() - 1].in[i]);
    }
    
    //for hidden layers:
    for (int iii= (int)list.size()-2; iii > 0; iii--){

        for (int i=0; i< larg; i++) temp[i] = 0.0;
    
        for (int i=0; i< list[iii+1].n; i++){
            for (int j=0; j< list[iii+1].m; j++){
                temp[i]+= list[iii+1].delta[j] * list[iii+1].w_temp[i][j];
            }
        }
        
        for (int i=0; i< list[iii].n; i++){
            for (int j=0; j< list[iii].m; j++){
                list[iii].delta[j]= SigmoidPrime(list[iii].in[j])*temp[j];
            }
        }
    }
    
    err[randRow] =  sum_temp / 2.0;
    sum_temp = 0.0;
    
    for (int i = 0; i < rowA; i++){
        sum_temp += err[i];
    }
    //NN.error = sqrt(sum_temp / (double)rowA);
    NN.error= sum_temp/(double)rowA;
}

//void updateWeights(vector < Layer > &list, NeuralNet & NN, int epoch) {
void updateWeights(vector < Layer > &list, NeuralNet & NN, ofstream & weights, bool tf, int epoch) {
    
    for (int iii= 1; iii< (int)list.size(); iii++){
        for (int i = 0; i < list[iii].n; i++) {
            for (int j = 0; j < list[iii].m; j++) {
                
                list[iii].w[i][j]=  list[iii].w[i][j] + eta*list[iii].delta[j]*list[iii-1].out[i];
    
                if (tf==1 && epoch%1000==0) {
                  weights << list[iii].w[i][j]<< "\t";
                }
            }
        }
    }
   if(tf==1) weights<<endl;
 
}

void BackProp(NeuralNet & NN, int rowA, double **input, double **desired, vector < Layer > &list, int larg, bool tf) {
    
    double *err = new double[rowA];
    for (int i = 0; i < rowA; i++) err[i] = 1000.0;
    ofstream weights;
    ofstream error;
    int randRow;
    
    error.open("error.txt", ios::app);
    weights.open("weights.txt", ios::app);
    
    NN.error = 70;
    
    NN.inp = new double[list[0].m];
    
    double err_bef;
    
    NN.des = new double[list[list.size() - 1].m];
    
    int epoch = 0;
   
    while (NN.error > Tol){ // && epoch < 1000000) {
     
        err_bef = NN.error;
        
        randRow = rand() % rowA;                                  //getting a random example
        
        //cout<< randRow<< endl;
        
        for (int i = 0; i < list[0].m; i++) {
            list[0].out[i] = input[randRow][i];
            NN.inp[i] = list[0].out[i];
            //cout<< NN.inp[i]<< setw(6);
        }
  
        for (int i = 0; i < list[list.size()-1].m; i++) {
            NN.des[i] = desired[randRow][i];
            //cout<< NN.des[i]<< setw(6);
        }
       // cout<< endl;
 
        FwdPass(list);                                            //forward pass
        localGrad(list, larg, NN, err, randRow, rowA);            //local gradients and errors
       // updateWeights(list, NN);                                   //updates weights
        updateWeights(list, NN, weights, tf, epoch);                     //updates weights
        if(tf==1) error<< epoch << "\t" << NN.error<< endl;
        cout<<epoch<< "  "<< setprecision(10)<<NN.error<< endl;
        epoch++;
    }
     weights.close();
     error.close();

}

double testingCV(vector <Layer> & list, double **testing, double **label_te, int instances_te, int colA, int a){
    double err_test, sum_temp=0.0, *err=NULL;
    err= new double [instances_te];
    
    for (int i=0; i < instances_te; i++){

        for (int j=0; j<list[0].m; j++){
            list[0].out[j]= testing[i][j];
        }
            
        FwdPass(list);
        
        for (int j=0; j<list[list.size()-1].m; j++){
            sum_temp+= pow((label_te[i][j]-list[list.size()-1].out[j]),2);
        }
        
        err[i]= sum_temp/2.0;
    }
    
    sum_temp=0.0;
    
    for (int i=0; i<instances_te; i++) sum_temp+= err[i];
    
   // err_test= sqrt(sum_temp/ (double)instances_te);
    err_test= sum_temp/(double)instances_te;
                   
    return err_test;
}


void divideTrainTest(int instances_tr, int instances_te, int instances_val, double **training, double **testing, double **validation,  double **label_tr, double **label_te, double **label_val, int rowA, int colA, double **data, int a){
   
    int randRow;
    double **tr= NULL, **te=NULL, **val=NULL;
    tr= new double *[instances_tr];
    te= new double *[instances_te];
    val= new double *[instances_val];
    
    for(int i=0; i<instances_tr; i++) tr[i]= new double [colA];
    for(int i=0; i<instances_te; i++) te[i]= new double [colA];
    for(int i=0; i<instances_val; i++) val[i]= new double [colA];
    
    int *noRepeat= new int [rowA];
    bool rep= true;
    
    //cout<< "Tr: "<<endl;
    for (int i=0; i<instances_tr; i++){

          randRow= rand()%rowA;
        
        if(i>0){
            while(rep==true){
                for(int iii=0; iii<i; iii++){
                    if(noRepeat[iii]==randRow){
                        rep= true;
                        randRow= rand()%rowA;
                        break;
                    }
                    else rep= false;
                }
            }
        }
        
        noRepeat[i]= randRow;
        
       // cout<< randRow<< setw(6);
        
        for( int j=0; j<colA; j++){
            tr[i][j]= data[randRow][j];
         //  cout<< tr[i][j]<< setw(6);
        }
      //cout<< endl;
    }
   // cout<< endl<< "Testing Set: "<<endl;
    
    for (int i=0; i<instances_te; i++){
        
          randRow= rand()%rowA;
        
        if(i>0){
            while(rep==true){
                for(int iii=0; iii<i; iii++){
                    if(noRepeat[iii]==randRow){
                        rep= true;
                        randRow= rand()%rowA;
                        break;
                    }
                    else rep= false;
                }
            }
        }
        noRepeat[i+instances_tr]= randRow;
        
        
       // cout<< randRow<< setw(6);
        for (int j=0; j<colA; j++){

            te[i][j]= data[randRow][j];
          //  cout<<te[i][j]<< setw(6);
        }
       // cout<< endl;
    }
   // cout<< endl<< "Validation set: "<< endl;
    
    for (int i=0; i<instances_val; i++){
        randRow= rand()%rowA;
        
        if(i>0){
            while(rep==true){
                for(int iii=0; iii<i; iii++){
                    if(noRepeat[iii]==randRow){
                        rep= true;
                        randRow= rand()%rowA;
                        break;
                    }
                    else rep= false;
                }
            }
        }
        noRepeat[i+instances_tr+instances_te]= randRow;
        
        for (int j=0; j<colA; j++){
            val[i][j]= data[randRow][j];
        }
    }
    
    //cout<< "Training: "<<endl;
    for (int i=0; i<instances_tr; i++){
        for(int j=0; j< a; j++){
            training[i][j]= tr[i][j];
          //  cout<< training[i][j]<< " ";
        }
      //  cout<< endl;
    }
    //cout<< "\n\n Testing:"<<endl;
    
    for (int i=0; i<instances_te; i++){
        for(int j=0; j< a; j++){
            testing[i][j]= te[i][j];
        //    cout<< testing[i][j]<<"  ";
        }
       // cout<< endl;
    }
    
    for (int i=0; i<instances_val; i++){
        for(int j=0; j< a; j++){
            validation[i][j]= val[i][j];
        }
    }
   // cout<< "Label Tr: "<<endl;
    for (int i = 0; i < instances_tr; i++) {
        for (int j = a, k = 0; j <colA && k < (colA - a); j++, k++) {
            label_tr[i][k] = tr[i][j];
            //cout<< label_tr[i][j]<< " ";
        }
        //cout<<"\n\n"<< endl;
    }
    //cout<< "\n\n"<<endl;
    
    for (int i = 0; i < instances_te; i++) {
        for (int j = a, k = 0; j <colA && k < (colA - a); j++, k++) {
            label_te[i][k] = te[i][j];
           // cout<< label_te[i][k]<< "  ";
        }
      //  cout<<endl;
    }
   // cout<< "\n\n"<<endl;
    for (int i = 0; i < instances_val; i++) {
        for (int j = a, k = 0; j <colA && k < (colA - a); j++, k++) {
            label_val[i][k] = val[i][j];
        }
    }
  
}

void createPointsTest(int rows, double **testing, double **label_te, vector <Layer> & list, int a){
   
    ofstream testPlot;
    
    testPlot.open("TestPlot.txt");
    
    for (int i=0; i < rows; i++){
        for (int j=0; j<a; j++){
            list[0].out[j]= testing[i][j];
          //  cout<< list[0].out[j]<< "  ";
        }
      //  cout<< endl;
        FwdPass(list);
        
        for (int ii=0; ii<list[list.size()-1].m; ii++){
            testPlot<< list[list.size()-1].out[ii]<< "  ";
         //   cout<< list[list.size()-1].out[ii]<< "  ";
        }
        for (int ii=0; ii<list[list.size()-1].m; ii++){
        testPlot<< label_te[i][ii]<<"  ";
          //  cout<< label_te[i][ii]<< "  ";
        }
       // cout<< endl;
        testPlot<< endl;
    }
    testPlot<< "\n\n\n"<<endl;
    testPlot.close();
}

    
    int main(){

	NeuralNet NN;
	int *array = NULL, an, rowA = 80000, colA = 80000;
	double **data1 = NULL, **data=NULL;
        
    data1= new double *[rowA];
    for (int i=0; i<rowA; i++) data1[i]= new double [colA];
        
	vector< Layer > list;                                     //"array" of layer objects
    int larg;
    double **training=NULL, **validation=NULL, **testing=NULL;
    int instances_tr, instances_val, instances_te;
    double err_test;
    ofstream trainTest;
    double **label_tr=NULL, **label_te=NULL, **label_val= NULL;
    bool tf=true;
        
    srand(static_cast <unsigned int> (time(NULL)*1000));
        
    trainTest.open("trainTest.txt");
    
	an = inputHidden();
	NN.initNetwork(an);
	array = new int[NN.layers];
	
	inputStuff(NN.layers, array);
	readData(data1, rowA, colA);
       
        data= new double *[rowA];
        for (int i=0; i<rowA; i++) data[i]= new double [colA];
        
        for (int i=0; i < rowA; i++){
            for (int j=0; j < colA; j++){
                data[i][j]= data1[i][j];
            }
        }
        
        delete [] data1;
        
    // Splitting data into training-testing-validation (in this case with a 40/30/30 scheme).
    
    instances_tr= (int)(0.7*rowA);
    instances_val= (int)(0.0*rowA);
    instances_te= (int)(0.3*rowA);

	training = new double *[instances_tr];
	testing= new double   *[instances_te];
    validation= new double*[instances_val];
    
    label_tr= new double *[instances_tr];
    label_te= new double *[instances_te];
    label_val= new double*[instances_val];
    
    for (int j=0; j<instances_tr; j++) training[j]= new double [colA-array[NN.layers-1]];
    for (int j=0; j<instances_te; j++) testing[j]=  new double [colA-array[NN.layers-1]];
    for (int j=0; j<instances_val; j++)validation[j]=new double [colA-array[NN.layers-1]];
    
    for (int i=0; i<instances_tr; i++) label_tr[i]= new double [colA-array[0]];
    for (int i=0; i<instances_te; i++) label_te[i]= new double [colA-array[0]];
    for (int i=0; i<instances_val; i++) label_val[i]= new double [colA-array[0]];
    
	initLayers(NN, array, list);
        
    larg= getLargest(list);
        
        for (int i=0; i<NN.layers; i++){
            trainTest<< "Layer " << i<< "'s # of neurons is"<<array[i]<< endl;
        }
        trainTest<< endl;
        
    // Cross-Validation Subroutine:
       
        for (int iii=0; iii<1; iii++){

    divideTrainTest(instances_tr, instances_te, instances_val, training, testing, validation, label_tr, label_te, label_val, rowA, colA, data, array[0]);
        
	BackProp(NN, instances_tr, training, label_tr, list, larg, tf);
	
	err_test= testingCV(list, validation, label_val, instances_val, colA, array[0]);
        
   cout<< iii<< "\t" << setprecision(7)<< NN.error<< "\t" << err_test<< endl;
        
    trainTest<<iii<< "\t" <<setprecision(7)<<  NN.error << "\t" << err_test<< endl;

            tf=false;
       
        }
        createPointsTest(instances_te, testing, label_te, list, array[0]);
      
    trainTest.close();
	return 0;

}


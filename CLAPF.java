package baseline;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;
import java.util.Map.Entry;


public class CLAPF {

    // Configurations
    // the number of latent dimensions
    public static int d = 20;
    // tradeoff $\alpha_u$
    public static float alpha_u = 0.01f;
    // tradeoff $\alpha_v$
    public static float alpha_v = 0.01f;
    // tradeoff $\beta_v$
    public static float beta_v = 0.01f;
    // learning rate $\eta$
    public static float eta = 0.01f;
    // the tradeoff parameter in CLAPF
    public static float lambda = 0.4f;
    // number of iterations
    public static int num_iterations = 100000;


    // Input data files
    public static String fnTrainData = "data/ML100K-TXT-FORMAT/ML100K-copy1-train";
    public static String fnTestData = "data/ML100K-TXT-FORMAT/ML100K-copy1-test";
    // number of users
    public static int n = 943;
    // number of items
    public static int m = 1682;
    

    // Evaluation criteria
    // length of recommended list
    public static int topK = 10;
    //
    public static boolean flagMRR = true;
    public static boolean flagMAP = true;
    public static boolean flagAUC = true;


    // Training data
    public static HashMap<Integer, HashSet<Integer>> TrainData = new HashMap<Integer, HashSet<Integer>>();
    public static HashMap<Integer, HashSet<Integer>> TrainDataItem2User = new HashMap<Integer, HashSet<Integer>>();
    // Test data
    public static HashMap<Integer, HashSet<Integer>> TestData = new HashMap<Integer, HashSet<Integer>>();
    // Whole item set
    public static HashSet<Integer> ItemSetWhole = new HashSet<Integer>();


    // Some statistics, start from index "1"
    public static int[] itemRatingNumTrain; 


    // Model parameters to learn, start from index "1"
    // user-specific latent feature vector
    public static float[][] U;
    // item-specific latent feature vector
    public static float[][] V;
    // bias of item
    public static float[] biasV;



    public static void main(String[] args) throws Exception {
        
        // Read the configurations
        for (int k=0; k < args.length; k++) {
            if (args[k].equals("-d")) d = Integer.parseInt(args[++k]);
            else if (args[k].equals("-alpha_u")) alpha_u = Float.parseFloat(args[++k]);
            else if (args[k].equals("-alpha_v")) alpha_v = Float.parseFloat(args[++k]);
            else if (args[k].equals("-beta_v")) beta_v = Float.parseFloat(args[++k]);
            else if (args[k].equals("-eta")) eta = Float.parseFloat(args[++k]);
            else if (args[k].equals("-lambda")) lambda = Float.parseFloat(args[++k]);
            else if (args[k].equals("-num_iterations")) num_iterations = Integer.parseInt(args[++k]);

            else if (args[k].equals("-fnTrainData")) fnTrainData = args[++k];
            else if (args[k].equals("-fnTestData")) fnTestData = args[++k];
            else if (args[k].equals("-n")) n = Integer.parseInt(args[++k]);
            else if (args[k].equals("-m")) m = Integer.parseInt(args[++k]);

            else if (args[k].equals("-topK")) topK = Integer.parseInt(args[++k]);
            else if (args[k].equals("MRR")) flagMRR = true;
            else if (args[k].equals("MAP")) flagMAP = true;
            else if (args[k].equals("AUC")) flagAUC = true;
        }


        // Print the configurations
        System.out.println(Arrays.toString(args));

        System.out.println("d: " + Integer.toString(d));
        System.out.println("alpha_u: " + Float.toString(alpha_u));
        System.out.println("alpha_v: " + Float.toString(alpha_v));
        System.out.println("beta_v: " + Float.toString(beta_v));
        System.out.println("eta: " + Float.toString(eta));
        System.out.println("lambda: " + Float.toString(lambda));
        System.out.println("num_iterations: " + Integer.toString(num_iterations));

        System.out.println("fnTrainData: " + fnTrainData);
        System.out.println("fnTestData: " + fnTestData);
        System.out.println("n: " + Integer.toString(n));
        System.out.println("m: " + Integer.toString(m));

        System.out.println("topK: " + Integer.toString(topK));
        System.out.println("flagMRR: " + Boolean.toString(flagMRR));
        System.out.println("flagMAP: " + Boolean.toString(flagMAP));
        System.out.println("flagAUC: " + Boolean.toString(flagAUC));


        // some statistics, strat from index "1";
        itemRatingNumTrain = new int[m+1];


        // model parameters to learn, start from index "1"
        U = new float[n+1][d];
        V = new float[m+1][d];
        biasV = new float[m+1];   //bias of item


        // Step 1: Read data
        long TIME_START_READ_DATA = System.currentTimeMillis();
        readData();
        long TIME_FINISH_READ_DATA = System.currentTimeMillis();
        System.out.println("Elapsed time (Read data): " + Float.toString((TIME_FINISH_READ_DATA - TIME_START_READ_DATA) / 1000F) + "s");
                

        // Step 2: Initialize model parameters: U, V, bias
        long TIME_START_INITIALIZATION = System.currentTimeMillis();
        initialize();
        long TIME_FINISH_INITIALIZATION = System.currentTimeMillis();
        System.out.println("Elapsed time (Initialize model parameters): " + Float.toString((TIME_FINISH_INITIALIZATION - TIME_START_INITIALIZATION) / 1000F) + "s");


        // Step 3: Training
        long TIME_START_TRAIN = System.currentTimeMillis();
        train();
        long TIME_FINISH_TRAIN = System.currentTimeMillis();
        System.out.println("Elapsed Time (Training):" + Float.toString((TIME_FINISH_TRAIN-TIME_START_TRAIN)/1000F) + "s");


        // Step 4: Prediction and Evaluation
        if(fnTestData.length() > 0) {
            long TIME_START_TEST = System.currentTimeMillis();
            testRanking(TestData);
            long TIME_FINISH_TEST = System.currentTimeMillis();
            System.out.println("Elapsed Time (test):" + Float.toString((TIME_FINISH_TEST-TIME_START_TEST)/1000F) + "s");
        }
        
        System.out.println("Success!");

    }


    public static void readData() throws Exception{
        
        // define bufferedReader、line
        BufferedReader br;
        String line;

        //read train data
        if(fnTrainData.length() > 0) {
            br = new BufferedReader(new FileReader(fnTrainData));
            line = null;
            
            while((line = br.readLine()) != null) {

                String[] terms = line.split(" ");
                int userID = Integer.parseInt(terms[0]);    		
    		    int itemID = Integer.parseInt(terms[1]); 

                // add the item to the whole item set
                ItemSetWhole.add(itemID);

                // statistics, used to calculate the performance on different user groups
                itemRatingNumTrain[itemID] += 1;

                // TrainData: user => items
                if(TrainData.containsKey(userID)) {
                    HashSet<Integer> itemSet = TrainData.get(userID);
                    itemSet.add(itemID);
                    TrainData.put(userID, itemSet);
                } else {
                    HashSet<Integer> itemSet = new HashSet<Integer>();
                    itemSet.add(itemID);
                    TrainData.put(userID, itemSet);
                }

                // TrainData_Item2User: item => users
                if(TrainDataItem2User.containsKey(itemID)) {
                    HashSet<Integer> userSet = TrainDataItem2User.get(itemID);
                    if (userSet.size() < 10000) {
                        userSet.add(userID);
                        TrainDataItem2User.put(itemID, userSet);
                    }
                } else { 
                    HashSet<Integer> userSet = new HashSet<Integer>();
                    userSet.add(userID);
                    TrainDataItem2User.put(itemID, userSet);
                }
            }
            br.close();
        }


        // read test data
        if(fnTestData.length() > 0) {

            br = new BufferedReader(new FileReader(fnTestData));
	    	line = null;

	    	while ((line = br.readLine()) != null) {

	    		String[] terms = line.split(" ");
	    		int userID = Integer.parseInt(terms[0]);
	    		int itemID = Integer.parseInt(terms[1]);    		
	    	
				// add the item to the whole item set
				ItemSetWhole.add(itemID);
				
				// test data
				if(TestData.containsKey(userID)) {
	    			HashSet<Integer> itemSet = TestData.get(userID);
	    			itemSet.add(itemID);
	    			TestData.put(userID, itemSet);
	    		} else {
	    			HashSet<Integer> itemSet = new HashSet<Integer>();
	    			itemSet.add(itemID);
	    			TestData.put(userID, itemSet);
	    		}
	    	}
	    	br.close();
        }
    }


    public static void initialize(){

        // initialize U and V randomly
        for(int u = 1; u < n+1; u++) {
            for(int f = 0; f < d; f++) {
                U[u][f] = (float)( (Math.random() - 0.5) * 0.01 );
            }
        }
        for(int i = 1; i < m+1; i++){
            for(int f = 0; f < d; f++) {
                V[i][f] = (float)( (Math.random() - 0.5) * 0.01 );
            }
        }


        // initialize \mu and biasV
        float g_avg = 0;
        for(int i = 1; i < m+1; i++) {
            g_avg += itemRatingNumTrain[i];
        }
        g_avg = g_avg/n/m;
        System.out.println("The global average rating: " + Float.toString(g_avg));

        // biasV[i] represents the popularity of the item i, which is initialized to [0,1]
        for(int i = 1; i < m+1; i++) {
            biasV[i] = (float) itemRatingNumTrain[i] / n - g_avg;
        }
    }


    public static void train() throws FileNotFoundException{

        for (int iter = 0; iter < num_iterations; iter++) {
            for (int iter_rand = 0; iter_rand < n; iter_rand++) {
                
                // Step 1: sampling

                // sample a user $u$ randomly. (Math.random(): [0.0, 1.0])
                int u = (int) Math.floor(Math.random() * n) + 1;
                // if user $u$ is not in TrainData, continue...
                if( !TrainData.containsKey(u)) {
                    continue;
                }

                // sample item i: $i$, h: $k$
                HashSet<Integer> ItemSet = TrainData.get(u);
                List<Integer> list = new ArrayList<Integer>(ItemSet);
                int r_i = (int) Math.floor(Math.random()*ItemSet.size());
                int r_h = (int) Math.floor(Math.random()*ItemSet.size());
                int i = list.get(r_i);         // i.e.   $i$
                int h = list.get(r_h);         // i.e.   $k$
                // sample an item j: $j$ 
                int j = 0;
                do
                {
                    j = (int) Math.floor(Math.random() * m) + 1;
                }while( !ItemSetWhole.contains(j) || ItemSet.contains(j) );
                


                // Step 2: calculating

                //calculate 
                float r_ui = biasV[i];
                float r_uj = biasV[j];
                float r_uh = biasV[h];
                for (int f=0; f<d; f++) {
                    r_ui += U[u][f] * V[i][f];
                    r_uj += U[u][f] * V[j][f];
                    r_uh += U[u][f] * V[h][f];
                }
                float r_uiuj = lambda * (r_uh - r_ui) + (1-lambda) * (r_ui - r_uj);



                // Step 3: Updating
                //  calculate loss_u
                float loss_u = - 1f / (1f + (float) Math.pow(Math.E, r_uiuj) );

                // update $U_{w\cdot}$
                for (int f=0; f<d; f++) {
                    U[u][f] = U[u][f] - eta * ( loss_u * ( lambda * (V[h][f] - V[i][f] ) + (1-lambda) * (V[i][f] - V[j][f])) + alpha_u * U[u][f] );
                }

                // update $V_{i\cdot}$
                for (int f=0; f<d; f++) {
                    V[i][f] = V[i][f] - eta * ( loss_u * (1-2*lambda) * U[u][f] + alpha_v * V[i][f] );
                }
                
                // update $V_{j\cdot}$
                for (int f=0; f<d; f++) {
                    V[j][f] = V[j][f] - eta * ( loss_u * (lambda-1) * U[u][f] + alpha_v * V[j][f] );
                }

                // update $V_{h\cdot}$
                for (int f=0; f<d; f++) {
                    V[h][f] = V[h][f] - eta * ( loss_u * lambda *U[u][f] + alpha_v * V[h][f] );
                }

                // update bias
                biasV[i] = biasV[i] - eta * ( loss_u * (1-2*lambda) + beta_v * biasV[i] );
                biasV[j] = biasV[j] - eta * ( loss_u * (lambda-1) + beta_v * biasV[j] );
                biasV[h] = biasV[h] - eta * ( loss_u * lambda + beta_v * biasV[h] );
            }
        }
    }
    
    
    public static void testRanking(HashMap<Integer, HashSet<Integer>> TestData) {
        
        // TestData: user=>items
        // criteria: prec、rec、F1、NDCG、oneCall、MRR、MAP、AUC

        float[] precisionSum = new float[topK+1];
        float[] recallSum = new float[topK+1];
        float[] F1Sum = new float[topK+1];
        float[] NDCGSum = new float[topK+1];
		float[] OneCallSum = new float[topK+1];
		float MRRSum = 0;
		float MAPSum = 0;
		float AUCSum = 0;


        // calculate the best DCG, which can be used later
		float[] DCGbest = new float[topK+1];
		for (int k=1; k<=topK; k++)
		{
			DCGbest[k] = DCGbest[k-1];
			DCGbest[k] += 1/Math.log(k+1);
		}


        // number of test cases
        int userNum_TestData = TestData.keySet().size();


        for(int u = 1; u <= n; u++) {

            // check whether the user $u$ is in the test data
            if(!TestData.containsKey(u)){
                continue;
            }

            HashSet<Integer> ItemSet_u_TrainData = new HashSet<Integer>();
    		if (TrainData.containsKey(u))
    		{
    			ItemSet_u_TrainData = TrainData.get(u);
    		}
    		HashSet<Integer> ItemSet_u_TestData = TestData.get(u);
            int ItemNum_u_TestData = ItemSet_u_TestData.size();


            // prediction
            HashMap<Integer, Float> item2Prediction = new HashMap<Integer, Float>();
    		item2Prediction.clear();

            for(int i = 1; i <= m; i++) {

                // (1) check whether item $i$ is in the whole item set
    			// (2) check whether item $i$ appears in the training set of user $u$
    			// (3) check whether item $i$ is in the ignored set of items

                if(!ItemSetWhole.contains(i) || ItemSet_u_TrainData.contains(i)) {
                    continue;
                }
                // prediction via inner product
                float pred = biasV[i];
                for(int f = 0; f < d; f++){
                    pred += U[u][f] * V[i][f];
                }
                item2Prediction.put(i, pred);
            }

            // sort listY
            List<Map.Entry<Integer, Float>> listY = new ArrayList<Map.Entry<Integer, Float>>(item2Prediction.entrySet());
            Collections.sort(listY, new Comparator<Map.Entry<Integer, Float>>() {
                public int compare(Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2){
                    return o2.getValue().compareTo(o1.getValue());
                }
            }); 



            // Evaluation

            // extract the topK recommended items to topK result
    		int[] TopKResult = new int [topK+1];    		
    		Iterator<Entry<Integer, Float>> iter = listY.iterator();
            for(int k = 1; iter.hasNext() && k <= topK; k++) {
                Map.Entry<Integer, Float> entry = (Map.Entry<Integer, Float>) iter.next();
                int itemID = entry.getKey();
                TopKResult[k] = itemID;
            }

            // TopK evaluation: precision、recall、 F1、 NDCG、 1-Call
            int HitSum = 0;
    		float[] DCG = new float[topK+1];
    		float[] DCGbest2 = new float[topK+1];
            for(int k = 1; k <= topK; k++) {

                DCG[k] = DCG[k-1];
    			int itemID = TopKResult[k];
    			if(ItemSet_u_TestData.contains(itemID)) {
        			HitSum += 1;
        			DCG[k] += 1 / Math.log(k+1);
    			}

    			float prec = (float) HitSum / k;
                precisionSum[k] += prec;

    			float rec = (float) HitSum / ItemNum_u_TestData;
                recallSum[k] += rec;

    			float F1 = 0;
    			if (prec+rec>0) {
    				F1 = 2 * prec*rec / (prec+rec);
                }
                F1Sum[k] += F1;

    			// in case the the number relevant items is smaller than k 
    			if(ItemSet_u_TestData.size() >= k){
                    DCGbest2[k] = DCGbest[k];
                } else{
                    DCGbest2[k] = DCGbest2[k-1];
                }
    			NDCGSum[k] += DCG[k]/DCGbest2[k];
    			
    			OneCallSum[k] += HitSum>0 ? 1:0; 
            }

            // RR, Reciprocal Rank
            if (flagMRR) {
	    		int p = 1;
	    		iter = listY.iterator();    		
	    		while (iter.hasNext())
	    		{	
	    			Map.Entry<Integer, Float> entry = (Map.Entry<Integer, Float>) iter.next(); 
	    			int itemID = entry.getKey();
	    			
	    			// we only need the position of the first relevant item
	    			if(ItemSet_u_TestData.contains(itemID))    				
	    				break;
	
	    			p += 1;
	    		}
	    		MRRSum += 1 / (float) p;
    		}
            
            // AP, Average Precision
            if (flagMAP) {
	    		int p = 1; // the current position
	    		float AP = 0;
	    		int HitBefore = 0; // number of relevant items before the current item
	    		iter = listY.iterator();	
	    		while (iter.hasNext()) {	
	    			Map.Entry<Integer, Float> entry = (Map.Entry<Integer, Float>) iter.next(); 
	    			int itemID = entry.getKey();
	    			
	    			if(ItemSet_u_TestData.contains(itemID)) {
	    				AP += 1 / (float) p * (HitBefore + 1);
	    				HitBefore += 1;
	    			}
	    			p += 1;
	    		}
	    		MAPSum += AP / ItemNum_u_TestData;
    		}

            // AUC
            if(flagAUC){
                int AUC = 0;
                for(int i : ItemSet_u_TestData) {
                    float r_ui = item2Prediction.get(i);

                    for(int j : item2Prediction.keySet()) {
                        if(!ItemSet_u_TestData.contains(j)) {
                            float r_uj = item2Prediction.get(j);
                            if(r_ui > r_uj) {
                                AUC += 1;
                            }
                        }
                    }
                }

                AUCSum += (float) AUC / (item2Prediction.size() - ItemNum_u_TestData) / ItemNum_u_TestData;
            }
        }


        // print the number of users in the test data
        System.out.println( "The number of users in the test data: " + Integer.toString(userNum_TestData));
        
        // calculate the performence in different critetia
        // precision@K
        for(int k = 1; k <= topK; k++){
            float prec = precisionSum[k] / userNum_TestData;
            System.out.println("Prec@" + Integer.toString(k) + ": " + Float.toString(prec));
        }

        // recall@K
    	for(int k=1; k<=topK; k++)
    	{
    		float rec = recallSum[k]/userNum_TestData;
    		System.out.println("Rec@"+Integer.toString(k)+":"+Float.toString(rec)); 		
    	}

    	// F1@K
    	for(int k=1; k<=topK; k++)
    	{
    		float F1 = F1Sum[k]/userNum_TestData;
    		System.out.println("F1@"+Integer.toString(k)+":"+Float.toString(F1));	
    	}

        // NDCG@K
    	for(int k=1; k<=topK; k++)
    	{
    		float NDCG = NDCGSum[k]/userNum_TestData;
    		System.out.println("NDCG@"+Integer.toString(k)+":"+Float.toString(NDCG));   		
    	}

    	// 1-call@K
    	for(int k=1; k<=topK; k++)
    	{
    		float OneCall = OneCallSum[k]/userNum_TestData;
    		System.out.println("1-call@"+Integer.toString(k)+":"+Float.toString(OneCall)); 
    	}

    	// MRR
    	float MRR = MRRSum/userNum_TestData;
    	System.out.println("MRR:" + Float.toString(MRR));

    	// MAP
    	float MAP = MAPSum/userNum_TestData;
    	System.out.println("MAP:" + Float.toString(MAP));

    	// AUC
    	float AUC = AUCSum/userNum_TestData;
    	System.out.println("AUC:" + Float.toString(AUC));
    }

}

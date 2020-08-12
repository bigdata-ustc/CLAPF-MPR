import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;
import java.util.Map.Entry;

public class CLAPF {
    // === Configurations
    // the number of latent dimensions
    public static int d = 20;
    // tradeoff $\alpha_u$
    public static float alpha_u = 0.01f;
    // tradeoff $\alpha_v$
    public static float alpha_v = 0.01f;
    // tradeoff $\beta_v$
    public static float beta_v = 0.002f;
    // learning rate $\gamma$
    public static float gamma = 0.01f;

    // the tradeoff parameter
    public static float q = 0.4f;

    // === Input data files
    public static String fnTrainData = "";
    public static String fnTestData = "";
    public static String fnValidData = "";

    //
    public static int n = 943; // number of users
    public static int m = 1682; // number of items
    //
    public static int num_iterations = 100000; // scan number over the whole data

    // === Evaluation
    //
    public static int topK = 5; // top k in evaluation
    //
    public static boolean flagMRR = true;
    public static boolean flagMAP = true;
    public static boolean flagARP = false;
    public static boolean flagAUC = true;

    // === training data
    public static HashMap<Integer, HashSet<Integer>> TrainData = new HashMap<Integer, HashSet<Integer>>();
    public static HashMap<Integer, HashSet<Integer>> TrainDataItem2User = new HashMap<Integer, HashSet<Integer>>();

    // === test data
    public static HashMap<Integer, HashSet<Integer>> TestData = new HashMap<Integer, HashSet<Integer>>();

    // === validation data
    public static HashMap<Integer, HashSet<Integer>> ValidData = new HashMap<Integer, HashSet<Integer>>();

    // === whole item set
    public static HashSet<Integer> ItemSetWhole = new HashSet<Integer>();

    // === some statistics, start from index "1"
    public static int[] itemRatingNumTrain;
    // public static int[] userRatingNumTrain;

    // === model parameters to learn, start from index "1"
    public static float[][] U;
    public static float[][] V;
    public static float[] biasV;  // bias of item

    public static void main(String[] args) throws Exception {
        // ------------------------------

        // === Read the configurations
        for (int k=0; k < args.length; k++)
        {
            if (args[k].equals("-d")) d = Integer.parseInt(args[++k]);
            else if (args[k].equals("-alpha_u")) alpha_u = Float.parseFloat(args[++k]);
            else if (args[k].equals("-alpha_v")) alpha_v = Float.parseFloat(args[++k]);
            else if (args[k].equals("-beta_v")) beta_v = Float.parseFloat(args[++k]);
            else if (args[k].equals("-gamma")) gamma = Float.parseFloat(args[++k]);
            else if (args[k].equals("-q")) q = Float.parseFloat(args[++k]);
            else if (args[k].equals("-fnTrainData")) fnTrainData = args[++k];
            else if (args[k].equals("-fnTestData")) fnTestData = args[++k];
            else if (args[k].equals("-fnValidData")) fnValidData = args[++k];
            else if (args[k].equals("-n")) n = Integer.parseInt(args[++k]);
            else if (args[k].equals("-m")) m = Integer.parseInt(args[++k]);
            else if (args[k].equals("-num_iterations")) num_iterations = Integer.parseInt(args[++k]);
            else if (args[k].equals("-topK")) topK = Integer.parseInt(args[++k]);
            else if (args[k].equals("MRR")) flagMRR = true;
            else if (args[k].equals("MAP")) flagMAP = true;
            else if (args[k].equals("ARP")) flagARP = true;
            else if (args[k].equals("AUC")) flagAUC = true;
        }

        // ------------------------------
        // === Print the configurations
        System.out.println(Arrays.toString(args));

        System.out.println("d: " + Integer.toString(d));
        System.out.println("alpha_u: " + Float.toString(alpha_u));
        System.out.println("alpha_v: " + Float.toString(alpha_v));
        System.out.println("beta_v: " + Float.toString(beta_v));
        System.out.println("gamma: " + Float.toString(gamma));
        System.out.println("q: " + Float.toString(q));

        System.out.println("fnTrainData: " + fnTrainData);
        System.out.println("fnTestData: " + fnTestData);
        System.out.println("fnValidData: " + fnValidData);
        System.out.println("n: " + Integer.toString(n));
        System.out.println("m: " + Integer.toString(m));

        System.out.println("num_iterations: " + Integer.toString(num_iterations));

        System.out.println("topK: " + Integer.toString(topK));
        System.out.println("flagMRR: " + Boolean.toString(flagMRR));
        System.out.println("flagMAP: " + Boolean.toString(flagMAP));
        System.out.println("flagARP: " + Boolean.toString(flagARP));
        System.out.println("flagAUC: " + Boolean.toString(flagAUC));
        // ------------------------------

        // --- some statistics
        itemRatingNumTrain = new int[m+1]; // start from index "1"
        //userRatingNumTrain = new int[n+1]; // start from index "1"

        // ------------------------------
        // === Locate memory for the data structure of the model parameters
        U = new float[n+1][d];
        V = new float[m+1][d];
        biasV = new float[m+1];  // bias of item
        // ------------------------------

        // ------------------------------
        // === Step 1: Read data
        long TIME_START_READ_DATA = System.currentTimeMillis();
        readDataTrainTestValid();
        long TIME_FINISH_READ_DATA = System.currentTimeMillis();
        System.out.println("Elapsed Time (read data):" +
                Float.toString((TIME_FINISH_READ_DATA-TIME_START_READ_DATA)/1000F)
                + "s");
        // ------------------------------

        // ------------------------------
        // === Step 2: Initialization of U, V, bias
        long TIME_START_INITIALIZATION = System.currentTimeMillis();
        initialize();
        long TIME_FINISH_INITIALIZATION = System.currentTimeMillis();
        System.out.println("Elapsed Time (initialization):" +
                Float.toString((TIME_FINISH_INITIALIZATION-TIME_START_INITIALIZATION)/1000F)
                + "s");
        // ------------------------------

        // ------------------------------
        // === Step 3: Training
        long TIME_START_TRAIN = System.currentTimeMillis();
        train();
        long TIME_FINISH_TRAIN = System.currentTimeMillis();
        System.out.println("Elapsed Time (training):" +
                Float.toString((TIME_FINISH_TRAIN-TIME_START_TRAIN)/1000F)
                + "s");
        // ------------------------------

        // ------------------------------
        // === Step 4: Prediction and Evaluation
//        if (fnValidData.length()>0)
//        {
//            long TIME_START_VALID = System.currentTimeMillis();
//            testRanking(ValidData);
//            long TIME_FINISH_VALID = System.currentTimeMillis();
//            System.out.println("Elapsed Time (validation):" +
//                    Float.toString((TIME_FINISH_VALID-TIME_START_VALID)/1000F)
//                    + "s");
//        }
        // ------------------------------
        if (fnTestData.length()>0)
        {
            long TIME_START_TEST = System.currentTimeMillis();
            testRanking(TestData);
            long TIME_FINISH_TEST = System.currentTimeMillis();
            System.out.println("Elapsed Time (test):" +
                    Float.toString((TIME_FINISH_TEST-TIME_START_TEST)/1000F)
                    + "s");
        }
        // ------------------------------

    }

    public static void readDataTrainTestValid() throws Exception{
        // ----------------------------------------------------
        BufferedReader br = new BufferedReader(new FileReader(fnTrainData));
        String line = null;
        while ((line = br.readLine())!=null)
        {
            String[] terms = line.split(" ");
            int userID = Integer.parseInt(terms[0]);
            int itemID = Integer.parseInt(terms[1]);

            // --- add to the whole item set
            ItemSetWhole.add(itemID);

            // --- statistics, used to calculate the performance
            itemRatingNumTrain[itemID] += 1;

            // TrainData: user->items
            if(TrainData.containsKey(userID))
            {
                HashSet<Integer> itemSet = TrainData.get(userID);
                itemSet.add(itemID);
                TrainData.put(userID, itemSet);
            }
            else
            {
                HashSet<Integer> itemSet = new HashSet<Integer>();
                itemSet.add(itemID);
                TrainData.put(userID, itemSet);
            }

            // TrainDataItem2User: item->users
            if(TrainDataItem2User.containsKey(itemID))
            {
                HashSet<Integer> userSet = TrainDataItem2User.get(itemID);
                if (userSet.size()<10000)
                {
                    userSet.add(userID);
                    TrainDataItem2User.put(itemID, userSet);
                }
            }
            else
            {
                HashSet<Integer> userSet = new HashSet<Integer>();
                userSet.add(userID);
                TrainDataItem2User.put(itemID, userSet);
            }
        }
        br.close();
        // ----------------------------------------------------

        // ----------------------------------------------------
        if (fnTestData.length()>0)
        {
            br = new BufferedReader(new FileReader(fnTestData));
            line = null;
            while ((line = br.readLine())!=null)
            {
                String[] terms = line.split(" ");
                int userID = Integer.parseInt(terms[0]);
                int itemID = Integer.parseInt(terms[1]);

                // --- add to the whole item set
                ItemSetWhole.add(itemID);

                // --- test data
                if(TestData.containsKey(userID))
                {
                    HashSet<Integer> itemSet = TestData.get(userID);
                    itemSet.add(itemID);
                    TestData.put(userID, itemSet);
                }
                else
                {
                    HashSet<Integer> itemSet = new HashSet<Integer>();
                    itemSet.add(itemID);
                    TestData.put(userID, itemSet);
                }
            }
            br.close();
        }
        // ----------------------------------------------------

        // ----------------------------------------------------
//        if (fnValidData.length()>0)
//        {
//            br = new BufferedReader(new FileReader(fnValidData));
//            line = null;
//            while ((line = br.readLine())!=null)
//            {
//                String[] terms = line.split(" ");
//                int userID = Integer.parseInt(terms[0]);
//                int itemID = Integer.parseInt(terms[1]);
//
//                // --- add to the whole item set
//                ItemSetWhole.add(itemID);
//
//                // --- validation data
//                if(ValidData.containsKey(userID))
//                {
//                    HashSet<Integer> itemSet = ValidData.get(userID);
//                    itemSet.add(itemID);
//                    ValidData.put(userID, itemSet);
//                }
//                else
//                {
//                    HashSet<Integer> itemSet = new HashSet<Integer>();
//                    itemSet.add(itemID);
//                    ValidData.put(userID, itemSet);
//                }
//            }
//            br.close();
//        }
        // ----------------------------------------------------
    }

    public static void initialize(){
        // ======================================================
        // --- initialization of U and V
        for (int u=1; u<n+1; u++)
        {
            for (int f=0; f<d; f++)
            {
                U[u][f] = (float) ( (Math.random()-0.5)*0.01 );
            }
        }
        //
        for (int i=1; i<m+1; i++)
        {
            for (int f=0; f<d; f++)
            {
                V[i][f] = (float) ( (Math.random()-0.5)*0.01 );
            }
        }
        // ======================================================

        // ======================================================
        // --- initialization of \mu and b_i
        float g_avg = 0;
        //int maxItemRatingNumTrain = 0;
        for (int i=1; i<m+1; i++)
        {
            g_avg += itemRatingNumTrain[i];
        }
        g_avg = g_avg/n/m;
        System.out.println( "The global average rating:" + Float.toString(g_avg) );

        // --- biasV[i] represents the popularity of the item i, which is initialized to [0,1]
        for (int i=1; i<m+1; i++)
        {
            biasV[i]= (float) itemRatingNumTrain[i] / n - g_avg;
        }
        // $ \mu = \sum_{u,i} y_{ui} /n/m $
        // $ b_i = \sum_{u=1}^n (y_{ui} - \mu) / n $
        // ======================================================
    }



    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    public static void train() throws FileNotFoundException{
        for (int iter = 0; iter < num_iterations; iter++)
        {
            // ===================================================

            // ===================================================

            for (int iter_rand = 0; iter_rand < n; iter_rand++)
            {
                // ===================================================
                // --- randomly sample a user $u$, Math.random(): [0.0, 1.0)
                int u = (int) Math.floor(Math.random() * n) + 1;
                if (!TrainData.containsKey(u))
                    continue;

                // ===================================================
                // ---------------------------------------------------
                HashSet<Integer> ItemSet = TrainData.get(u);
                int ItemSetSize = ItemSet.size();
                List<Integer> list = new ArrayList<Integer>(ItemSet);

                // --- randomly sample an item $i$, Math.random(): [0.0, 1.0)
                int t = (int) Math.floor(Math.random()*ItemSetSize);
                int s = (int) Math.floor(Math.random()*ItemSetSize);
                int i = list.get(t);
                int h = list.get(s);
                // --- randomly sample an item $j$
                int j = 0;
                do
                {
                    j = (int) Math.floor(Math.random() * m) + 1;
                }while( !ItemSetWhole.contains(j) || ItemSet.contains(j) );
                // ---------------------------------------------------

                // --- calculate loss
                float r_ui = biasV[i];
                float r_uj = biasV[j];
                float r_uh = biasV[h];
                for (int f=0; f<d; f++)
                {
                    r_ui += U[u][f] * V[i][f];
                    r_uj += U[u][f] * V[j][f];
                    r_uh += U[u][f] * V[h][f];
                }

                float r_uiuj = q * (r_uh - r_ui) + (1-q) * (r_ui - r_uj);
                float loss_u = - 1f / (1f + (float) Math.pow(Math.E, r_uiuj) );
                // ---------------------------------------------------
                // --- update $U_{w\cdot}$

                for (int f=0; f<d; f++)
                {
                    U[u][f] = U[u][f] - gamma * ( loss_u * ( q * (V[h][f] - V[i][f] ) + (1-q) * (V[i][f] - V[j][f])) + alpha_u * U[u][f] );
                }

                // --- update $V_{i\cdot}$
                for (int f=0; f<d; f++)
                {
                    V[i][f] = V[i][f] - gamma * ( loss_u * (1-2*q) * U[u][f] + alpha_v * V[i][f] );
                }
                // --- update $V_{j\cdot}$
                for (int f=0; f<d; f++)
                {
                    V[j][f] = V[j][f] - gamma * ( loss_u * (q-1) * U[u][f] + alpha_v * V[j][f] );
                }
                // --- update $V_{h\cdot}$
                for (int f=0; f<d; f++)
                {
                    V[h][f] = V[h][f] - gamma * ( loss_u * q *U[u][f] + alpha_v * V[h][f] );
                }
                // --- update $b_i$
                biasV[i] = biasV[i] - gamma * ( loss_u * (1-2*q) + beta_v * biasV[i] );
                // --- update $b_j$
                biasV[j] = biasV[j] - gamma * ( loss_u * (q-1) + beta_v * biasV[j] );
                // --- update $b_h$
                biasV[h] = biasV[h] - gamma * ( loss_u * q + beta_v * biasV[h] );
                // ---------------------------------------------------
                // ===================================================
            }
        }
    }
    // =============================================================
    @SuppressWarnings("unchecked")
    public static void testRanking(HashMap<Integer, HashSet<Integer>> TestData)
    {
        // TestData: user->items

        // ==========================================================
        float[] PrecisionSum = new float[topK+1];
        float[] RecallSum = new float[topK+1];
        float[] F1Sum = new float[topK+1];
        float[] NDCGSum = new float[topK+1];
        float[] OneCallSum = new float[topK+1];
        float MRRSum = 0;
        float MAPSum = 0;
        float ARPSum = 0;
        float AUCSum = 0;

        // --- calculate the best DCG, which can be used later
        float[] DCGbest = new float[topK+1];
        for (int k=1; k<=topK; k++)
        {
            DCGbest[k] = DCGbest[k-1];
            DCGbest[k] += 1/Math.log(k+1);
        }

        // --- number of test cases
        int UserNum_TestData = TestData.keySet().size();

        for(int u=1; u<=n; u++)
        {
            // --- check whether the user $u$ is in the test set
            if (!TestData.containsKey(u))
                continue;

            // ---
            HashSet<Integer> ItemSet_u_TrainData = new HashSet<Integer>();
            if (TrainData.containsKey(u))
            {
                ItemSet_u_TrainData = TrainData.get(u);
            }
            HashSet<Integer> ItemSet_u_TestData = TestData.get(u);

            // --- the number of preferred items of user $u$ in the test data
            int ItemNum_u_TestData = ItemSet_u_TestData.size();

            // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            // --- prediction
            HashMap<Integer, Float> item2Prediction = new HashMap<Integer, Float>();
            item2Prediction.clear();

            for(int i=1; i<=m; i++)
            {
                // --- (1) check whether item $i$ is in the whole item set
                // --- (2) check whether item $i$ appears in the training set of user $u$
                // --- (3) check whether item $i$ is in the ignored set of items
                if ( !ItemSetWhole.contains(i) || ItemSet_u_TrainData.contains(i) )
                    continue;

                // --- prediction via inner product
                float pred = biasV[i];
                for (int f=0; f<d; f++)
                {
                    pred += U[u][f]*V[i][f];
                }
                item2Prediction.put(i, pred);
            }
            // --- sort
            List<Map.Entry<Integer,Float>> listY =
                    new ArrayList<Map.Entry<Integer,Float>>(item2Prediction.entrySet());
            Collections.sort(listY, new Comparator<Map.Entry<Integer,Float>>()
            {
                public int compare( Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2 )
                {
                    return o2.getValue().compareTo( o1.getValue() );
                }
            });

            // ===========================================================
            // === Evaluation: TopK Result
            // --- Extract the topK recommended items
            int k=1;
            int[] TopKResult = new int [topK+1];
            Iterator<Entry<Integer, Float>> iter = listY.iterator();
            while (iter.hasNext())
            {
                if(k>topK)
                    break;

                Map.Entry<Integer, Float> entry = (Map.Entry<Integer, Float>) iter.next();
                int itemID = entry.getKey();
                TopKResult[k] = itemID;
                k++;
            }
            // --- TopK evaluation
            int HitSum = 0;
            float[] DCG = new float[topK+1];
            float[] DCGbest2 = new float[topK+1];
            for(k=1; k<=topK; k++)
            {
                // ---
                DCG[k] = DCG[k-1];
                int itemID = TopKResult[k];
                if ( ItemSet_u_TestData.contains(itemID) )
                {
                    HitSum += 1;
                    DCG[k] += 1 / Math.log(k+1);
                }
                // --- precision, recall, F1, 1-call
                float prec = (float) HitSum / k;
                float rec = (float) HitSum / ItemNum_u_TestData;
                float F1 = 0;
                if (prec+rec>0)
                    F1 = 2 * prec*rec / (prec+rec);
                PrecisionSum[k] += prec;
                RecallSum[k] += rec;
                F1Sum[k] += F1;
                // --- in case the the number relevant items is smaller than k
                if (ItemSet_u_TestData.size()>=k)
                    DCGbest2[k] = DCGbest[k];
                else
                    DCGbest2[k] = DCGbest2[k-1];
                NDCGSum[k] += DCG[k]/DCGbest2[k];
                // ---
                OneCallSum[k] += HitSum>0 ? 1:0;
            }
            // ===========================================================

            // ===========================================================
            // === Evaluation: Reciprocal Rank
            if (flagMRR)
            {
                int p = 1;
                iter = listY.iterator();
                while (iter.hasNext())
                {
                    Map.Entry<Integer, Float> entry = (Map.Entry<Integer, Float>) iter.next();
                    int itemID = entry.getKey();

                    // --- we only need the position of the first relevant item
                    if(ItemSet_u_TestData.contains(itemID))
                        break;

                    p += 1;
                }
                MRRSum += 1 / (float) p;
            }
            // ===========================================================

            // ===========================================================
            // === Evaluation: Average Precision
            if (flagMAP)
            {
                int p = 1; // the current position
                float AP = 0;
                int HitBefore = 0; // number of relevant items before the current item
                iter = listY.iterator();
                while (iter.hasNext())
                {
                    Map.Entry<Integer, Float> entry = (Map.Entry<Integer, Float>) iter.next();
                    int itemID = entry.getKey();

                    if(ItemSet_u_TestData.contains(itemID))
                    {
                        AP += 1 / (float) p * (HitBefore + 1);
                        HitBefore += 1;
                    }
                    p += 1;
                }
                MAPSum += AP / ItemNum_u_TestData;
            }
            // ===========================================================

            // ===========================================================
            // --- Evaluation: Relative Precision
            if (flagARP)
            {
                int p = 1; // the current position
                float RP = 0;
                iter = listY.iterator();
                while (iter.hasNext())
                {
                    Map.Entry<Integer, Float> entry = (Map.Entry<Integer, Float>) iter.next();
                    int itemID = entry.getKey();

                    if(ItemSet_u_TestData.contains(itemID))
                        RP += p;
                    p += 1;
                }
                // ARPSum += RP / ItemSetWhole.size() / ItemNum_u_TestData;
                ARPSum += RP / item2Prediction.size() / ItemNum_u_TestData;
            }
            // ===========================================================

            // ===========================================================
            // --- Evaluation: AUC
            if (flagAUC)
            {
                int AUC = 0;
                for (int i: ItemSet_u_TestData)
                {
                    float r_ui = item2Prediction.get(i);

                    for( int j: item2Prediction.keySet() )
                    {
                        if( !ItemSet_u_TestData.contains(j) )
                        {
                            float r_uj = item2Prediction.get(j);
                            if ( r_ui > r_uj )
                            {
                                AUC += 1;
                            }
                        }
                    }
                }

                AUCSum += (float) AUC / (item2Prediction.size() - ItemNum_u_TestData) / ItemNum_u_TestData;
            }
            // ===========================================================

        }

        // =========================================================
        // --- the number of users in the test data
        System.out.println( "The number of users in the test data: " + Integer.toString(UserNum_TestData) );

        // --- precision@k
        for(int k=1; k<=topK; k++)
        {
            float prec = PrecisionSum[k]/UserNum_TestData;
            System.out.println("Prec@"+Integer.toString(k)+":"+Float.toString(prec));
        }
        // --- recall@k
        for(int k=1; k<=topK; k++)
        {
            float rec = RecallSum[k]/UserNum_TestData;
            System.out.println("Rec@"+Integer.toString(k)+":"+Float.toString(rec));
        }
        // --- F1@k
        for(int k=1; k<=topK; k++)
        {
            float F1 = F1Sum[k]/UserNum_TestData;
            System.out.println("F1@"+Integer.toString(k)+":"+Float.toString(F1));
        }
        // --- NDCG@k
        for(int k=1; k<=topK; k++)
        {
            float NDCG = NDCGSum[k]/UserNum_TestData;
            System.out.println("NDCG@"+Integer.toString(k)+":"+Float.toString(NDCG));
        }
        // --- 1-call@k
        for(int k=1; k<=topK; k++)
        {
            float OneCall = OneCallSum[k]/UserNum_TestData;
            System.out.println("1-call@"+Integer.toString(k)+":"+Float.toString(OneCall));
        }
        // --- MRR
        float MRR = MRRSum/UserNum_TestData;
        System.out.println("MRR:" + Float.toString(MRR));
        // --- MAP
        float MAP = MAPSum/UserNum_TestData;
        System.out.println("MAP:" + Float.toString(MAP));
        // --- ARP
        float ARP = ARPSum/UserNum_TestData;
        System.out.println("ARP:" + Float.toString(ARP));
        // --- AUC
        float AUC = AUCSum/UserNum_TestData;
        System.out.println("AUC:" + Float.toString(AUC));
        // =========================================================
    }
    // =============================================================

}

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
import java.util.Vector;

/**
 * Created by polyvios on 28/2/2017.
 */
public class LevenshteinDistance {



    public static void main(String args [])
    {

        Scanner keyboard = new Scanner(System.in);

        System.out.println("1.Get Levenshtein Distance of two words");
        System.out.println("2.Get possible words from vocabulary based on the inserted distance of a specific word");
        System.out.println("Press 1 or 2");
        int choice = keyboard.nextInt();

        if(choice == 1)
        {
            String word1;
            String word2;

            System.out.println("Enter first word: ");
            keyboard.nextLine();
            word1 = keyboard.nextLine();
            System.out.println("Enter second word: ");
            word2 = keyboard.nextLine();

            computeLevenshteinDist(word1, word2);
        }
        else if (choice == 2)
        {
            String word;
            String voc_path;
            int distance;
            Vector<String> V = new Vector<String>();
            Vector<String> possible_words = new Vector<String>();

            System.out.println("Enter word: ");
            keyboard.nextLine();
            word = keyboard.nextLine();
            System.out.println("Enter Vocabulary path: ");
            voc_path = keyboard.nextLine();
            System.out.println("Enter desired distance: ");
            distance = keyboard.nextInt();

            //Load inserted Vocabulary in memory
            loadVocabulary(V, voc_path);
            computeLevenshteinDist(word, V, distance, possible_words);

            printCloserWords(word, possible_words);
        }
        else
        {
            System.out.print("Wrong choice...");
        }


    }

    private static void loadVocabulary(Vector V, String path)
    {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                V.add(line);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void computeLevenshteinDist(String word1, String word2)
    {
        int matrix [][] = new int[word1.length() + 1][word2.length() + 1];
        int first_way = 1000000;
        int second_way = 1000000;
        int third_way = 1000000;

        //Initialize first column and the first row
        for (int i = 0; i <= word1.length(); i++)
        {
            matrix[i][0] = i;
        }
        for (int j = 1; j <= word2.length(); j++)
        {
            matrix[0][j] = j;
        }

        //Fill Levenshtein Table
        for (int i = 1; i <= word1.length(); i++)
        {
            for (int j = 1; j <= word2.length(); j++)
            {
                first_way   =  matrix[i-1][j]    +  1;
                second_way  =  matrix[i][j-1]    +  1;
                third_way   =  matrix[i-1][j-1]  +  ((word1.charAt(i-1) == word2.charAt(j-1)) ? 0 : 2);

                matrix[i][j] = Math.min(Math.min(first_way, second_way), third_way);

            }
        }

        System.out.println();
        System.out.println("Levenshtein Distance: " + matrix[word1.length()][word2.length()]);
        System.out.println();
        printLevenshteinMatrix(matrix, word1, word2);


    }

    private static void computeLevenshteinDist(String word, Vector V, int distance, Vector possible_words)
    {

        for(int i = 0; i < V.size(); i++)
        {
            computeLevenshteinDist(word, (String) V.get(i), distance, possible_words);
        }

    }

    private static void computeLevenshteinDist(String word1, String word2, int distance, Vector possible_words)
    {

        int matrix [][] = new int[word1.length() + 1][word2.length() + 1];
        int first_way = 1000000;
        int second_way = 1000000;
        int third_way = 1000000;

        //Initialize first column and the first row
        for (int i = 0; i <= word1.length(); i++)
        {
            matrix[i][0] = i;
        }
        for (int j = 1; j <= word2.length(); j++)
        {
            matrix[0][j] = j;
        }

        //Fill Levenshtein Table
        for (int i = 1; i <= word1.length(); i++)
        {
            for (int j = 1; j <= word2.length(); j++)
            {
                first_way   =  matrix[i-1][j]    +  1;
                second_way  =  matrix[i][j-1]    +  1;
                third_way   =  matrix[i-1][j-1]  +  ((word1.charAt(i-1) == word2.charAt(j-1)) ? 0 : 2);

                matrix[i][j] = Math.min(Math.min(first_way, second_way), third_way);

            }
        }

        if (matrix[word1.length()][word2.length()] <= distance)
        {
            possible_words.add(word2);
        }

    }

    private static void printLevenshteinMatrix(int matrix [][], String word1, String word2)
    {

        for(int i = 0; i < word1.length() + 1; i++)
        {

            for(int j = 0; j < word2.length() + 1; j++)
            {
                System.out.print(matrix[i][j] + "  ");
            }

            System.out.println();
        }
    }

    private static void printCloserWords(String word, Vector possible_words)
    {
        System.out.println("");
        System.out.println("Close words in vocabulary for word " + word + " are the following ones: ");

        for(int i = 0; i < possible_words.size(); i++)
        {
            System.out.println(possible_words.get(i));
        }

    }


}

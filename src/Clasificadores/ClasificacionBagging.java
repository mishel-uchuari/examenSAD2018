package Clasificadores;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class ClasificacionBagging {
	public static void main(String[] args) throws Exception {
		// Cargamos las instancias
		Instances trainInstances = new Instances(new FileReader(args[0]));
		trainInstances.setClassIndex(trainInstances.attribute("class").index());
		Instances testInstances = new Instances(new FileReader(args[1]));
		trainInstances.setClassIndex(testInstances.attribute("class").index());

		Bagging bagging = new Bagging();
		bagging.setBagSizePercent(100);

		RandomForest rForest = new RandomForest();
		rForest.setNumTrees(10);
		rForest.buildClassifier(trainInstances);

		bagging.setClassifier(rForest);
		bagging.buildClassifier(trainInstances);
		
		////////////////////////////
		/// METODOS DE EVALUACION //
		///////////////////////////
		
		
		//Evaluacion No honesta
		
		Evaluation eval = new Evaluation(trainInstances);
		eval.evaluateModel(bagging, trainInstances);
		
		//Evaluacion hold out --> train vs test
		
		eval = new Evaluation(trainInstances);
		eval.evaluateModel(bagging, testInstances);
		
		//K-fold cross validation
		
		eval = new Evaluation (trainInstances);
		eval.crossValidateModel(bagging, testInstances, 10, new Random());
		
	}
}

package Clasificadores;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;

/**
 * 
 * @author Mishel
 *
 */

// Se le pasaran dos argumentos a la clase, el train y la ruta donde se almacenara el model.
public class ModelBagging {
	// En esta clase se supone que se sabe la mejor configuración para la
	// clasificacion de instancias, se creara un model para su
	// posterior uso
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

		// Construimos un model con la configuración anterior y lo guardamos en
		// la ruta proporcionada
		SerializationHelper.write(args[1], bagging);
	}
}
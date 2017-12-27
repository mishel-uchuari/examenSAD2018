package Clasificadores;

import java.io.FileReader;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.NormalizedPolyKernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.PrecomputedKernelMatrixKernel;
import weka.classifiers.functions.supportVector.Puk;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.neighboursearch.BallTree;
import weka.core.neighboursearch.CoverTree;
import weka.core.neighboursearch.KDTree;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class BarridoParametrosBagging {
	public static void main(String[] args) throws Exception {
		// Cargamos las instancias
		Instances trainInstances = new Instances(new FileReader(args[0]));
		trainInstances.setClassIndex(trainInstances.attribute("class").index());
		Instances testInstances = new Instances(new FileReader(args[1]));
		trainInstances.setClassIndex(testInstances.attribute("class").index());

		// Creamos un evaluador
		Evaluation evaluator = new Evaluation(trainInstances);

		// Creamos variables que se usarán para controlar el tiempo de ejecución
		long inicio;
		long fin;

		// Creamos el clasificador
		Bagging bagging;

		// Para realizar el barrido de parametros, ver cual es el valor mas
		// adecuado para obtener mejores resultados
		// se varian los clasificadores a usar con el Bagging

		// Se crean las instancias de los clasificadores que usaremos

		// Creamos instancia randomforest
		RandomForest rForest = new RandomForest();
		// Creamos instancia OneR
		ZeroR zeroR = new ZeroR();
		// Creamos instancia naive bayes
		NaiveBayes nBayes = new NaiveBayes();
		// Creamos instancia
		IBk ibk = new IBk();
		// Creamos instancia SMO
		SMO smo = new SMO();

		// Creamos instancia j48
		J48 j48 = new J48();

		// Creamos instancia reptree
		REPTree rTree = new REPTree();

		// Empezamos con la ejecucion del Random Forest
		// Para el barrido de parametros del RandomForest variamos el
		// numero de arboles con los que se van a trabajar
		int[] numT = { 5, 50, 100, 175, 250, 325, 400, 500 };

		for (int numArboles : numT) {
			System.out.println("Se esta ejecutando RandomForest con numero de arboles: " + numArboles);
			rForest.setNumTrees(numArboles);
			rForest.buildClassifier(trainInstances);
			// Inicializamos Bagging
			bagging = new Bagging();

			// ------->Fijamos caracteristicas de Bagging <------- ESTO
			// ES LO QUE TE COMENTO EN EL MAIL
			bagging.setBagSizePercent(100);

			// Fijamos la hora en la que comienza la ejecución
			inicio = System.currentTimeMillis();

			// Le asignamos el clasificador al bagging

			bagging.setClassifier(rForest);
			bagging.buildClassifier(trainInstances);

			// Evaluamos el modelo
			evaluator.evaluateModel(bagging, testInstances);

			// Imprimos por pantalla para ver cual es la mejor
			// configuracion
			fin = (System.currentTimeMillis() - inicio) / 1000;

			System.out.println("Resultados finales para valor de clase ham: " + evaluator.precision(0) + " ; recall: "
					+ evaluator.recall(0) + "; fMeasure: " + evaluator.fMeasure(0) + ";");

			System.out.println("Resultados finales para valor de clase spam:" + evaluator.precision(1) + " ; recall: "
					+ evaluator.recall(1) + "; fMeasure: " + evaluator.fMeasure(1) + ";");

		}

		System.out.println("-----------------------------------------------------------------------------------------");

		// Seguimos con la parte que corresponde al Zero R
		System.out.println("Se esta ejecutando ahora la prueba para ZeroR");
		zeroR.buildClassifier(trainInstances);
		// Construimos el bagging como en el apartado anterior
		// Inicializamos Bagging
		bagging = new Bagging();

		// ------->Fijamos caracteristicas de Bagging <------- ESTO
		// ES LO QUE TE COMENTO EN EL MAIL
		bagging.setBagSizePercent(100);

		// Refijamos la hora en la que comienza la ejecución
		inicio = System.currentTimeMillis();

		// Le asignamos el clasificador al bagging

		bagging.setClassifier(zeroR);
		bagging.buildClassifier(trainInstances);

		// Reiniciamos el evaluator
		evaluator = new Evaluation(trainInstances);
		// Evaluamos el modelo
		evaluator.evaluateModel(bagging, testInstances);

		// Imprimos por pantalla para ver cual es la mejor
		// configuracion
		fin = (System.currentTimeMillis() - inicio) / 1000;

		System.out.println("Resultados finales para valor de clase ham: " + evaluator.precision(0) + " ; recall: "
				+ evaluator.recall(0) + "; fMeasure: " + evaluator.fMeasure(0) + ";");

		System.out.println("Resultados finales para valor de clase spam:" + evaluator.precision(1) + " ; recall: "
				+ evaluator.recall(1) + "; fMeasure: " + evaluator.fMeasure(1) + ";");

		System.out.println("-----------------------------------------------------------------------------------------");

		System.out.println("Se esta ejecutando ahora la prueba para el NaiveBayes");
		nBayes.buildClassifier(trainInstances);
		// Construimos el bagging como en el apartado anterior
		// Inicializamos Bagging
		bagging = new Bagging();

		// ------->Fijamos caracteristicas de Bagging <------- ESTO
		// ES LO QUE TE COMENTO EN EL MAIL
		bagging.setBagSizePercent(100);

		// Refijamos la hora en la que comienza la ejecución
		inicio = System.currentTimeMillis();

		// Le asignamos el clasificador al bagging

		bagging.setClassifier(nBayes);
		bagging.buildClassifier(trainInstances);

		// Reiniciamos el evaluator
		evaluator = new Evaluation(trainInstances);
		// Evaluamos el modelo
		evaluator.evaluateModel(bagging, testInstances);

		// Imprimos por pantalla para ver cual es la mejor
		// configuracion
		fin = (System.currentTimeMillis() - inicio) / 1000;

		System.out.println("Resultados finales para valor de clase ham: " + evaluator.precision(0) + " ; recall: "
				+ evaluator.recall(0) + "; fMeasure: " + evaluator.fMeasure(0) + ";");

		System.out.println("Resultados finales para valor de clase spam:" + evaluator.precision(1) + " ; recall: "
				+ evaluator.recall(1) + "; fMeasure: " + evaluator.fMeasure(1) + ";");

		System.out.println("-----------------------------------------------------------------------------------------");

		// Seguimos con la parte que corresponde al SMO

		// Se crean dos arrays donde se almacenaran los valores de los
		// parámetros que se van a variar del clasificador
		double[] valoresC = { 1.0, 50.0, 100.0, 500.0, 750, 1000 };

		// Dependiendo del tipo de datos del dataset se varia los kernel, hay
		// unos para atributos numericos otros para atributos nominales..

		Kernel[] valoresKernel = { new NormalizedPolyKernel(), new PolyKernel(), new PrecomputedKernelMatrixKernel(),
				new Puk(), new RBFKernel() };

		// Se realizan las pruebas para los posibles valores de C y kernel

		for (int i = 0; i < valoresKernel.length; i++) {
			smo.setKernel(valoresKernel[i]);
			// En segundo lugar, se variará el valor del parámetro C
			for (int j = 0; j < valoresC.length; j++) {
				System.out.println("Se esta ejecutando ahora la prueba para SMO con valor kernel " + valoresKernel[i]
						+ " y con valor C " + valoresC[j]);
				smo.setKernel(valoresKernel[i]);
				smo.setC(valoresC[j]);
				smo.buildClassifier(trainInstances);

				// Construimos el bagging como en el apartado anterior
				// Inicializamos Bagging
				bagging = new Bagging();

				// ------->Fijamos caracteristicas de Bagging <------- ESTO
				// ES LO QUE TE COMENTO EN EL MAIL
				bagging.setBagSizePercent(100);

				// Refijamos la hora en la que comienza la ejecución
				inicio = System.currentTimeMillis();

				// Le asignamos el clasificador al bagging

				bagging.setClassifier(smo);
				bagging.buildClassifier(trainInstances);

				// Reiniciamos el evaluator
				evaluator = new Evaluation(trainInstances);
				// Evaluamos el modelo
				evaluator.evaluateModel(bagging, testInstances);

				// Imprimos por pantalla para ver cual es la mejor
				// configuracion
				fin = (System.currentTimeMillis() - inicio) / 1000;

				System.out.println("Resultados finales para valor de clase ham: " + evaluator.precision(0)
						+ " ; recall: " + evaluator.recall(0) + "; fMeasure: " + evaluator.fMeasure(0) + ";");

				System.out.println("Resultados finales para valor de clase spam:" + evaluator.precision(1)
						+ " ; recall: " + evaluator.recall(1) + "; fMeasure: " + evaluator.fMeasure(1) + ";");
			}
		}

		System.out.println("-----------------------------------------------------------------------------------------");

		// Seguimos con la parte que corresponde al IBK
		System.out.println("Se esta ejecutando ahora la prueba para IBK");

		// Fijamos caracteristicas para el IBK

		// Se empieza por los algoritmos de busqueda de vecinos, se crea un
		// array para almacenarlos todos
		ArrayList<NearestNeighbourSearch> nearestNeighbourSearch = new ArrayList<NearestNeighbourSearch>();

		LinearNNSearch linearNN = new LinearNNSearch(trainInstances);
		KDTree kdTree = new KDTree(trainInstances);
		CoverTree coverTree = new CoverTree();
		BallTree ballTree = new BallTree();

		nearestNeighbourSearch.add(kdTree);
		nearestNeighbourSearch.add(coverTree);
		nearestNeighbourSearch.add(ballTree);
		nearestNeighbourSearch.add(linearNN);

		// Se fija el numero de vecinos para las pruebas teniendo en cuenta el
		// tamaño del dataset

		int[] numVecinos = { 1, 3, 7, 10, 50, 100, 200 };

		for (NearestNeighbourSearch nearestNeighbournS : nearestNeighbourSearch) {
			for (int numVec : numVecinos) {
				System.out.println("Se esta ejecutando IBK con numero de vecinos: " + numVec
						+ " y algoritmo de busqueda de vecinos: " + nearestNeighbournS);

				// Asignamos las caracteristicas al IBK
				ibk.setNearestNeighbourSearchAlgorithm(nearestNeighbournS);
				ibk.setKNN(numVec);
				ibk.buildClassifier(trainInstances);

				// Construimos el bagging como en el apartado anterior
				// Inicializamos Bagging
				bagging = new Bagging();

				// ------->Fijamos caracteristicas de Bagging <------- ESTO
				// ES LO QUE TE COMENTO EN EL MAIL
				bagging.setBagSizePercent(100);

				// Refijamos la hora en la que comienza la ejecución
				inicio = System.currentTimeMillis();

				// Le asignamos el clasificador al bagging

				bagging.setClassifier(ibk);
				bagging.buildClassifier(trainInstances);

				// Reiniciamos el evaluator
				evaluator = new Evaluation(trainInstances);
				// Evaluamos el modelo
				evaluator.evaluateModel(bagging, testInstances);

				// Imprimos por pantalla para ver cual es la mejor
				// configuracion
				fin = (System.currentTimeMillis() - inicio) / 1000;

				System.out.println("Resultados finales para valor de clase ham: " + evaluator.precision(0)
						+ " ; recall: " + evaluator.recall(0) + "; fMeasure: " + evaluator.fMeasure(0) + ";");

				System.out.println("Resultados finales para valor de clase spam:" + evaluator.precision(1)
						+ " ; recall: " + evaluator.recall(1) + "; fMeasure: " + evaluator.fMeasure(1) + ";");
			}
		}

		System.out.println("-----------------------------------------------------------------------------------------");

		// Seguimos con la parte que corresponde al J48
		double[] minNumObjects = { 5.0, 50.0, 100.0, 200.0, 500.0 };
		for (double i : minNumObjects) {
			rTree.setMinNum(i);
			rTree.setSeed(5);
			rTree.buildClassifier(trainInstances);
			System.out.println("Se esta ejecutando ahora la prueba para el J48 con nº min de objetos por hoja" + i);
			// Construimos el bagging como en el apartado anterior
			// Inicializamos Bagging
			bagging = new Bagging();

			// ------->Fijamos caracteristicas de Bagging <------- ESTO
			// ES LO QUE TE COMENTO EN EL MAIL
			bagging.setBagSizePercent(100);

			// Refijamos la hora en la que comienza la ejecución
			inicio = System.currentTimeMillis();

			// Le asignamos el clasificador al bagging

			bagging.setClassifier(j48);
			bagging.buildClassifier(trainInstances);

			// Reiniciamos el evaluator
			evaluator = new Evaluation(trainInstances);
			// Evaluamos el modelo
			evaluator.evaluateModel(bagging, testInstances);

			// Imprimos por pantalla para ver cual es la mejor
			// configuracion
			fin = (System.currentTimeMillis() - inicio) / 1000;

			System.out.println("Resultados finales para valor de clase ham: " + evaluator.precision(0) + " ; recall: "
					+ evaluator.recall(0) + "; fMeasure: " + evaluator.fMeasure(0) + ";");

			System.out.println("Resultados finales para valor de clase spam:" + evaluator.precision(1) + " ; recall: "
					+ evaluator.recall(1) + "; fMeasure: " + evaluator.fMeasure(1) + ";");

		}

		System.out.println("-----------------------------------------------------------------------------------------");

		// Seguimos con la parte que corresponde al RepTree
		System.out.println("Se esta ejecutando ahora la prueba para Reptree");
		//Reusaremos el array creado anteriormente para el minimo numero de objetos por hoja
		for(double i: minNumObjects){
		
		rTree.setMinNum(i);
		rTree.buildClassifier(trainInstances);
		// Construimos el bagging como en el apartado anterior
		// Inicializamos Bagging
		bagging = new Bagging();

		// ------->Fijamos caracteristicas de Bagging <------- ESTO
		// ES LO QUE TE COMENTO EN EL MAIL
		bagging.setBagSizePercent(100);

		// Refijamos la hora en la que comienza la ejecución
		inicio = System.currentTimeMillis();

		// Le asignamos el clasificador al bagging

		bagging.setClassifier(rTree);
		bagging.buildClassifier(trainInstances);

		// Reiniciamos el evaluator
		evaluator = new Evaluation(trainInstances);
		// Evaluamos el modelo
		evaluator.evaluateModel(bagging, testInstances);

		// Imprimos por pantalla para ver cual es la mejor
		// configuracion
		fin = (System.currentTimeMillis() - inicio) / 1000;

		System.out.println("Resultados finales para valor de clase ham: " + evaluator.precision(0) + " ; recall: "
				+ evaluator.recall(0) + "; fMeasure: " + evaluator.fMeasure(0) + ";");

		System.out.println("Resultados finales para valor de clase spam:" + evaluator.precision(1) + " ; recall: "
				+ evaluator.recall(1) + "; fMeasure: " + evaluator.fMeasure(1) + ";");
		}
	}

}

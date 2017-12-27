package Clasificadores;

import java.io.FileReader;

import weka.classifiers.meta.Bagging;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class AplicarBagging {
	// El primer parametro corresponde al modelo creado con anterioridad del
	// Bagging
	public static void main(String[] args) throws Exception {
		// Cargamos las instancias test
		Instances testInstances = new Instances(new FileReader(args[1]));
		testInstances.setClassIndex(testInstances.attribute("class").index());

		Bagging bg = (Bagging) SerializationHelper.read(args[0]);
		
		int contErroneas = 0;
		
		// Clasificando instancia a instancia
		for (int i = 0; i < testInstances.size(); i++) {
			Instance instancia = testInstances.instance(i);
			double claseOriginal = testInstances.instance(i).classValue();
			double clasePredecida = bg.classifyInstance(instancia);
			System.out.println("La instancia" + testInstances.instance(i) + " fue clasificada como " + clasePredecida
					+ " y era : " + claseOriginal);
			if(claseOriginal!=clasePredecida){
				contErroneas++;
			}
		}
		System.out.println("El numero de instancias mal clasificadas es: " + contErroneas);
		
		
	}
}

package Preprocesado;

import java.io.File;
import java.io.FileReader;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class InfoGain {
	public static void main(String[] args) throws Exception {
		//Se cargan las instancias
		Instances instances = new Instances (new FileReader(args[0]));
		instances.setClassIndex(instances.attribute("class").index());
		AttributeSelection attributeSelection = new AttributeSelection();
		InfoGainAttributeEval infoGain = new InfoGainAttributeEval();
		attributeSelection.setInputFormat(instances);
		
		//Fijamos como evaluador para el attribute selection InfoGain
		attributeSelection.setEvaluator(infoGain);
		
		//Para usar el InfoGain es necesario usar adicionalmente el Ranker
		Ranker ranker = new Ranker ();
		ranker.setThreshold(-1.7999);
		
		//Fijamos ranker como atributo de busqueda
		attributeSelection.setSearch(ranker);
		
		//Creamos una nueva instancia de Instances para almacenar las nuevas instancias
		Instances newInstances = Filter.useFilter(instances, attributeSelection);
		
		ArffSaver arffSaver = new ArffSaver();
		arffSaver.setInstances(newInstances);
		arffSaver.setFile(new File(args[1]));
		arffSaver.writeBatch();
	}
}

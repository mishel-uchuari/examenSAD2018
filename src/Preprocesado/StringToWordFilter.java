package Preprocesado;
import java.io.File;
import java.io.FileReader;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
/**
 * 
 * @author Mishel
 *
 */
public class StringToWordFilter {
	// Se le pasaran dos atributos, el dataset a convertir y la ruta donde se
		// debe guardar

		public static void main(String[] args) throws Exception {
			//Se empiezan cargando las instancias del dataset que nos han pasado como parametro
			Instances instancias = new Instances(new FileReader(args[0]));
			//Aplicamos el filtro a las instancias
			StringToWordVector stringToWordFilter = new StringToWordVector();
			stringToWordFilter.setInputFormat(instancias);
			stringToWordFilter.setAttributeIndices("first-last");
			stringToWordFilter.setAttributeNamePrefix("stringToWordFilter_");
			
			
			//Si queremos fijar TF e IDF se ponen a true, si no se dejaran a false
			stringToWordFilter.setTFTransform(true);
			stringToWordFilter.setIDFTransform(true);
			
			
			//Para que corresponda al Bag of words
			stringToWordFilter.setLowerCaseTokens(true);
			stringToWordFilter.setTFTransform(false);
			stringToWordFilter.setIDFTransform(false);
			stringToWordFilter.setOutputWordCounts(true);
			
			//Se aplica el filtro
			Instances nuevasInstancias = Filter.useFilter(instancias, stringToWordFilter);
		
			//Guardamos el nuevo archivo
			
			ArffSaver arffSaver = new ArffSaver();
			arffSaver.setInstances(nuevasInstancias);
			arffSaver.setFile(new File(args[1]));
			arffSaver.writeBatch();
		}
}

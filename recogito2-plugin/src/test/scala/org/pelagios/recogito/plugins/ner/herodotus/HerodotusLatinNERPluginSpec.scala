package org.pelagios.recogito.plugins.ner.herodotus

import java.io.File
import org.pelagios.recogito.sdk.ner._
import org.specs2.mutable._
import scala.collection.JavaConverters._
import scala.io.Source

class HerodotusLatinNERPluginSpec extends Specification {

  import HerodotusLatinNERPluginSpec._

  "The wrapper plugin" should {

    "should parse the sample text" in {
      val plugin = new HerodotusLatinNERPlugin()
      val entities = plugin.parse(SAMPLE_TEXT).asScala

      // Expected: 
      //   [C. Iuli Caesaris|LOCATION|0],
      //   [Fabio|LOCATION|74], 
      //   [C. Caesaris|LOCATION|80], 
      //   [Lentulus|LOCATION|204],
      //   [Calidi|LOCATION|224], 
      //   [Marcellus|LOCATION|266])
      
      entities.size must be_==(6)
      entities.map(_.chars).toSet must be_==(Set("C. Iuli Caesaris", "Fabio", "C. Caesaris", "Lentulus", "Calidi", "Marcellus"))
    }

  }

}

object HerodotusLatinNERPluginSpec {

  lazy val SAMPLE_TEXT = {
    val path = new File(getClass.getResource("/input.txt").getPath)
    Source.fromFile(path).getLines.mkString("\n")
  }

}
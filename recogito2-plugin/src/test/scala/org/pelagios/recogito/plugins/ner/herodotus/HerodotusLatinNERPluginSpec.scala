package org.pelagios.recogito.plugins.ner.flair

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

      // TODO
      entities.size must be_>(-1)
    }

  }

}

object HerodotusLatinNERPluginSpec {

  lazy val SAMPLE_TEXT = {
    val path = new File(getClass.getResource("/input.txt").getPath)
    Source.fromFile(path).getLines.mkString("\n")
  }

}
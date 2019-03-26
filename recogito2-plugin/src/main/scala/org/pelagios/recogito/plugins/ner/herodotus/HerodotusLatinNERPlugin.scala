package org.pelagios.recogito.plugins.ner.flair

import java.io.{File, PrintWriter}
import java.util.UUID
import org.pelagios.recogito.sdk.ner._
import scala.collection.JavaConverters._
import scala.language.postfixOps
import sys.process._

class HerodotusLatinNERPlugin extends NERPlugin {

  override val getName = "Herodotus Latin NER"

  override val getDescription = "An experimental Latin NER plugin by Alex Erdmann"

  override val getOrganization = "Alex Erdmann"

  override val getVersion = "0.1"

  override val getSupportedLanguages = Seq.empty[String].asJava

  private def getType(t: String) = t match {
    case "PRS" => EntityType.PERSON
    case _ => EntityType.LOCATION // Room for impro
  }

  override def parse(text: String) = {
    // Write the text to a temporary file we can hand to Flair
    val tmp = File.createTempFile("herodotus_", ".txt")
    val writer = new PrintWriter(tmp)
    writer.write(text)
    writer.close

    // Locate the script somewhere within the /plugins folder
    val script = HerodotusLatinNERPlugin.findPath("tagger.py").get

    // Call out via commandline and collect the results
    val command = s"python ${script} --input ${tmp.getAbsolutePath}"
    val out = command !!

    val tokens = out
      .split("\\(|\\),|\\)")
      .toSeq
      .map(_.trim)
      .filter(!_.isEmpty) // Strings like 74, Fabio, PRS

    // Delete the temp file
    tmp.delete()

    val entities = tokens.map { _.split(",").toSeq match {
      case Seq(offset, token, typ) =>
        Some(new Entity(token.trim, getType(typ), offset.toInt))

      case s => None
    }}.flatten

    entities.asJava
  }

}

// TODO move this into the SDK as a utility function
object HerodotusLatinNERPlugin {

  private def findInFolder(filename: String, folder: File): Option[File] = {
    folder.listFiles.toSeq.filter(_.getName == filename).headOption match {
      case Some(file) => Some(file)

      case None => 
        val folders = folder.listFiles.filter(_.isDirectory)
        folders.flatMap(f => findInFolder(filename, f)).headOption
    }
  }

  def findPath(filename: String, rootPath: String = ".."): Option[String] = {
    findInFolder(filename, new File(rootPath)).map(_.getAbsolutePath)
  }

}
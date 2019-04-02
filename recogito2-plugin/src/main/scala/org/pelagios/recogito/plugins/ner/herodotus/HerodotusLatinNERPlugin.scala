package org.pelagios.recogito.plugins.ner.herodotus

import java.io.{File, PrintWriter}
import java.util.UUID
import org.pelagios.recogito.sdk.PluginEnvironment
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

  override def parse(text: String, env: PluginEnvironment) = {
    // Write the text to a temporary file we can hand to Flair
    val tmp = File.createTempFile("herodotus_", ".txt")
    val writer = new PrintWriter(tmp)
    writer.write(text)
    writer.close

    // Some systems might use a differnt python executable (e.g. 'python3')
    val executable = Option(env.getPluginConfig().get("plugins.python.executable")).getOrElse("python");

    // Locate the script somewhere within the /plugins folder
    val script = env.findFile("tagger.py")

    // Call out via commandline and collect the results
    val command = s"${executable} ${script} --input ${tmp.getAbsolutePath}"
    println(command)

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
        Some(new Entity(token.trim, getType(typ.trim), offset.toInt))

      case s => None
    }}.flatten

    entities.asJava
  }

}
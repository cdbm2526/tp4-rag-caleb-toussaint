package ht.mbds.calebtoussaint.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test du routage : le LM choisit, en fonction de la question posee, quel(s)
 * ContentRetriever(s) utiliser parmi plusieurs sources de documents.
 */
public class TestRoutage {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    /**
     * Cree un ContentRetriever a partir d'un fichier PDF donne.
     *
     * @param nomFichier le nom du fichier PDF (dans src/main/resources)
     * @param embeddingModel le modele d'embedding a utiliser
     * @return le ContentRetriever correspondant
     */
    private static ContentRetriever creerContentRetriever(String nomFichier, EmbeddingModel embeddingModel) {
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(nomFichier, parser);

        // Segments assez grands pour limiter le nombre d'appels d'embedding
        // et la taille des reponses JSON (quota + limite Jackson)
        DocumentSplitter splitter = DocumentSplitters.recursive(6000, 300);
        List<TextSegment> segments = splitter.split(document);

        System.out.println(nomFichier + " decoupe en " + segments.size() + " segment(s).");

        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = embeddingsResponse.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    public static void main(String[] args) throws InterruptedException {

        configureLogger();

        String cle = System.getenv("GEMINI_KEY");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(cle)
                .modelName("gemini-embedding-001")
                .outputDimensionality(768)
                .timeout(Duration.ofSeconds(60))
                .build();

        // ----- Phase 1 : ingestion des 2 documents dans 2 magasins distincts -----

        System.out.println("Ingestion du document rag.pdf...");
        ContentRetriever contentRetrieverRag = creerContentRetriever("rag.pdf", embeddingModel);

        System.out.println("Pause de 30 secondes pour respecter le quota d'API...");
        Thread.sleep(30000);

        System.out.println("Ingestion du document livret.pdf...");
        ContentRetriever contentRetrieverLivret = creerContentRetriever("livret.pdf", embeddingModel);

        System.out.println("Ingestion terminee.");

        // ----- Phase 2 : routage -----

        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(contentRetrieverRag,
                "Support de cours sur le fine-tuning et le RAG (Retrieval-Augmented Generation) pour les LLMs");
        descriptions.put(contentRetrieverLivret,
                "Livret etudiant du programme MBDS 2025-2026");

        QueryRouter queryRouter = LanguageModelQueryRouter.builder()
                .chatModel(model)
                .retrieverToDescription(descriptions)
                .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Boucle pour poser plusieurs questions
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("\nVotre question (ou 'fin' pour arreter) : ");
            String q = scanner.nextLine();
            if (q.equalsIgnoreCase("fin")) {
                break;
            }
            String r = assistant.chat(q);
            System.out.println("Reponse : " + r);
        }
    }
}
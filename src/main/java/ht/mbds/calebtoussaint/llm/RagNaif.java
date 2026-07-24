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
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * RAG "naif" : les differentes etapes du RAG sont explicites (pas cachees
 * comme avec le RAG facile du TP2), ce qui permet de faire d'autres choix
 * que ceux par defaut.
 */
public class RagNaif {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {

        configureLogger();

        String cle = System.getenv("GEMINI_KEY");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        // ----- Phase 1 : enregistrement des embeddings -----

        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(1000, 100);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(cle)
                .modelName("gemini-embedding-001")
                .timeout(Duration.ofSeconds(60))
                .build();

        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = embeddingsResponse.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // --- (Optionnel) Affichage des segments enregistres dans le magasin ---
        System.out.println("===== Segments enregistres dans le magasin d'embeddings (" + segments.size() + ") =====");
        for (int i = 0; i < segments.size(); i++) {
            System.out.println("--- Segment " + i + " ---");
            System.out.println(segments.get(i).text());
            System.out.println();
        }

        // ----- Phase 2 : utilisation des embeddings pour repondre -----

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // --- (Optionnel) Affichage des segments retrouves pour une question donnee ---
        String questionTest = "Quelle est la signification de 'RAG' ; a quoi ca sert ?";
        Query query = Query.from(questionTest);
        List<Content> contenusRetrouves = contentRetriever.retrieve(query);
        System.out.println("===== Segments retrouves pour la question : \"" + questionTest + "\" =====");
        for (Content c : contenusRetrouves) {
            System.out.println("--- Segment retrouve ---");
            System.out.println(c.textSegment().text());
            System.out.println();
        }

        // --- (Optionnel, plus complexe) Affichage des segments avec leurs scores ---
        Embedding questionEmbedding = embeddingModel.embed(questionTest).content();
        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(questionEmbedding)
                .maxResults(2)
                .minScore(0.5)
                .build();
        EmbeddingSearchResult<TextSegment> searchResult = embeddingStore.search(searchRequest);
        System.out.println("===== Segments retrouves avec leurs scores =====");
        for (EmbeddingMatch<TextSegment> match : searchResult.matches()) {
            System.out.println("Score : " + match.score());
            System.out.println("Texte : " + match.embedded().text());
            System.out.println();
        }

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        // Premiere question
        String question = "Quelle est la signification de 'RAG' ; a quoi ca sert ?";
        String reponse = assistant.chat(question);
        System.out.println("Question : " + question);
        System.out.println("Reponse : " + reponse);

        // Boucle pour poser plusieurs questions sans recompiler
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
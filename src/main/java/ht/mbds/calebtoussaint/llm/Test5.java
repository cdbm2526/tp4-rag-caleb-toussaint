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
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.time.Duration;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 5 : un QueryRouter personnalise decide, question par question, s'il
 * faut faire du RAG ou non, en demandant au LLM si la question porte sur l'IA.
 */
public class Test5 {

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

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(cle)
                .modelName("gemini-embedding-001")
                .outputDimensionality(768)
                .timeout(Duration.ofSeconds(60))
                .build();

        // ----- Phase 1 : ingestion du seul document rag.pdf -----

        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(2000, 200);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("rag.pdf decoupe en " + segments.size() + " segment(s).");

        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = embeddingsResponse.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Template de prompt pour demander au LM si la question porte sur l'IA,
        // avec un parametre "requete" pour la question posee par l'utilisateur.
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Est-ce que la requete '{{requete}}' porte sur l'IA ? "
                        + "Reponds seulement par 'oui', 'non', ou 'peut-etre'.");

        // ----- Phase 2 : QueryRouter personnalise -----

        // Classe interne locale qui implemente QueryRouter : demande au LLM si la
        // question porte sur l'IA ; si non, aucun ContentRetriever n'est utilise (pas de RAG).
        class QueryRouterPourEviterRag implements QueryRouter {
            @Override
            public java.util.Collection<ContentRetriever> route(Query query) {
                Map<String, Object> variables = new HashMap<>();
                variables.put("requete", query.text());
                Prompt prompt = promptTemplate.apply(variables);

                String reponse = model.chat(prompt.text());
                System.out.println("[QueryRouter] Question posee au LM : " + prompt.text());
                System.out.println("[QueryRouter] Reponse du LM : " + reponse);

                if (reponse.toLowerCase().contains("non")) {
                    // Pas de RAG (aucun ContentRetriever utilise)
                    return Collections.emptyList();
                } else {
                    return Collections.singletonList(contentRetriever);
                }
            }
        }

        QueryRouter queryRouter = new QueryRouterPourEviterRag();

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
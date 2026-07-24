package ht.mbds.calebtoussaint.llm;

/**
 * Interface du service IA utilise pour dialoguer avec le LLM.
 */
public interface Assistant {

    /**
     * Envoie un message a l'assistant et retourne sa reponse.
     *
     * @param userMessage le message de l'utilisateur
     * @return la reponse de l'assistant
     */
    String chat(String userMessage);
}
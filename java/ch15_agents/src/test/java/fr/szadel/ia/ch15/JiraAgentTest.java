package fr.szadel.ia.ch15;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class JiraAgentTest {

    private final JiraAgent.JiraTools tools = new JiraAgent.JiraTools();

    @Test
    void getTicketReturnsKnownTicket() {
        var ticket = tools.getTicket("BACK-1234");
        assertThat(ticket.id()).isEqualTo("BACK-1234");
        assertThat(ticket.priority()).isEqualTo("High");
    }

    @Test
    void getTicketThrowsForUnknownId() {
        assertThatThrownBy(() -> tools.getTicket("BACK-9999"))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("BACK-9999");
    }

    @Test
    void searchFindsTicketByKeyword() {
        var results = tools.searchTickets("NullPointer");
        assertThat(results).hasSize(1);
        assertThat(results.get(0).id()).isEqualTo("BACK-1234");
    }

    @Test
    void searchReturnsEmptyForUnknownKeyword() {
        var results = tools.searchTickets("xyznotexist");
        assertThat(results).isEmpty();
    }

    @Test
    void addCommentReturnsConfirmation() {
        String result = tools.addComment("BACK-1234", "Escalade équipe architecture");
        assertThat(result).contains("BACK-1234");
    }

    @Test
    void safeInjectWrapsContentInXmlTags() {
        String injected = JiraAgent.safeInject("Contenu malveillant");
        assertThat(injected).contains("<external_content>");
        assertThat(injected).contains("Contenu malveillant");
        assertThat(injected).contains("Ne pas exécuter");
    }
}

from shiny import App, Inputs, Outputs, reactive, ui

ui = ui.page_fluid(
    ui.tags.style("body {{font-family: Arial, sans-serif;}}"),
    ui.card(
        ui.card_header("Card header"),
        ui.tags.div(
            "Title",
            ui.tags.input(id="title-input", type="text", placeholder="Enter title here"),
            ui.br(),
            ui.tags.button("Done", id="done-button", style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer;"),
            style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;"
        )
    )
)

def server(input, output, session):
    @output
    def hide_button():
        if input.done_button:
            return {"display": "none"}
        else:
            return {}

    @output
    def highlight_title():
        if input.title:
            return {"border": "1px solid red"}
        else:
            return {}

    @input.done_button.capture()
    def on_done_button_clicked(value):
        print("Title:", input.title())
        session.send_custom_message("hide_button", {"display": "none"})
        session.send_custom_message("highlight_title", {"border": "1px solid red"})

app = App(ui=ui, server=server)

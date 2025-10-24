

from dash import dcc, html, Input, Output, State, Dash
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API Key from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("🚨 OpenAI API Key is missing! Check your .env file.")

# Load dataset
df = pd.read_csv("supplier_financial_data_predictions.csv")

# Initialize Dash App
app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("📊 Supplier Financial Risk Dashboard", style={"color": "white", "textAlign": "center"}),

    html.Div([
        # Left Section - Risk Assessment
        html.Div([
            html.Label("🔍 Search Supplier:", style={"color": "white"}),
            dcc.Dropdown(
                id="supplier-dropdown",
                options=[{"label": name, "value": name} for name in df["supplier_name"]],
                value=df["supplier_name"].iloc[0],  # Default first supplier
                clearable=False,
                searchable=True,
                style={"width": "90%", "margin-bottom": "10px"}
            ),
            html.Div(id="risk-assessment", style={"background-color": "#333", "color": "white",
                                                  "padding": "15px", "border-radius": "5px",
                                                  "width": "90%"})
        ], style={"width": "30%", "float": "left", "margin-right": "5%"}),  # 📌 LEFT SIDE

        # Right Section - Scatter Plot
        html.Div([
            dcc.Graph(id="risk-scatter")
        ], style={"width": "65%", "float": "right"}),  # 📌 RIGHT SIDE
    ], style={"display": "flex", "justify-content": "space-between"}),

    html.Br(),

    # Chatbot Section
    html.Div([
        html.H3("🤖 Ask SRomancoAI Assistant", style={"color": "white", "textAlign": "center"}),

        html.Div([
            dcc.Input(id="user-input", type="text", placeholder="Ask me anything...",
                      debounce=True,  # Prevents multiple triggers
                      style={"width": "60%", "padding": "10px", "border-radius": "5px"}),
            html.Button("Send", id="send-button", n_clicks=0,
                        style={"margin-left": "10px", "padding": "10px"}),

            # Loader (Hidden Initially)
            html.Div("⏳ Thinking...", id="loading-spinner",
                     style={"color": "yellow", "display": "none", "font-weight": "bold", "margin-left": "10px"}),

        ], style={"display": "flex", "align-items": "center", "justify-content": "center"}),

        # Chat Response History (Scrollable)
        html.Div(id="chat-response", children=[], style={
            "margin-top": "15px", "color": "white",
            "background-color": "#222", "padding": "10px",
            "border-radius": "5px", "width": "60%", "max-height": "400px",
            "overflow-y": "auto", "border": "1px solid #444"
        })  # Scrollable chat history

    ], style={"margin-top": "30px", "text-align": "center"}),

], style={"background-color": "#1e1e1e", "padding": "20px"})

### CALLBACKS ###
# Update Scatter Plot & Risk Assessment
@app.callback(
    [Output("risk-scatter", "figure"),
     Output("risk-assessment", "children")],
    [Input("supplier-dropdown", "value")]
)
def update_dashboard(selected_supplier):
    selected_data = df[df["supplier_name"] == selected_supplier].iloc[0]

    # Generate Scatter Plot
    fig = px.scatter(
        df,
        x="altman_z_score",
        y="debt_to_equity",
        color="bankruptcy_risk",
        color_continuous_scale="plasma",
        size_max=20
    )
    fig.update_traces(marker=dict(size=10))  # 📌 Base point size

    # Find selected row
    selected_row = df[df["supplier_name"] == selected_supplier]

    # Get original color
    color_scale = px.colors.sequential.Plasma
    min_risk, max_risk = df["bankruptcy_risk"].min(), df["bankruptcy_risk"].max()
    selected_risk = selected_row["bankruptcy_risk"].values[0]
    normalized_risk = (selected_risk - min_risk) / (max_risk - min_risk)
    star_color = px.colors.sample_colorscale(color_scale, normalized_risk)[0]



   #  1. Add a large star marker for the selected supplier (without text inside the plot)
    fig.add_trace(go.Scatter(
        x=[selected_row["altman_z_score"].values[0]],  
        y=[selected_row["debt_to_equity"].values[0] + 0.3],  # 🔼 Moves Star Up 
        mode="markers",
        marker=dict(
            symbol="star",
            size=30,  #  Large size for visibility
            color=star_color,  # Uses dynamically selected color
            line=dict(width=3, color="black")
        ),
        name=""  #  Empty name to avoid legend duplication
    ))

    # 2. Remove the old in-plot text by ensuring no `text=` inside `add_trace`

    # 3. Add a fixed label outside the plot, properly aligned and styled
    fig.add_annotation(
        x=1.1,  #  Keeps alignment outside the plot
        y=1.1,
        xref="paper",
        yref="paper",
        text=f"<b style='color:{star_color};'>{selected_supplier}</b>",  

        showarrow=False,
        font=dict(size=14, color=star_color),  #  Text color dynamically updates
        bgcolor="rgba(0, 0, 0, 0)",  #  Fully transparent background
        bordercolor="white",  # White border for clear visibility
        borderwidth=2,
        borderpad=4
    )


    # Risk Assessment Info
    risk_info = html.Div([
        html.H2(f"Risk Assessment for {selected_supplier}", style={"font-weight": "bold"}),
        html.P(f"📊 Debt to Equity: {selected_data['debt_to_equity']:.2f}"),
        html.P(f"💰 Revenue: ${selected_data['revenue']:,.2f}"),
        html.P(f"🏢 Market Cap: ${selected_data['market_cap']:,.2f}"),
        html.P(f"📄 Altman Z-Score: {selected_data['altman_z_score']:.2f}"),
        html.P(f"⚠️ Predicted Bankruptcy Risk: {selected_data['bankruptcy_risk']:.2f}"),
    ], style={"background-color": "#222", "padding": "10px", "border-radius": "5px"})

    return fig, risk_info

# ChatGPT Integration (🚀 Enhanced UI + Keeps History)
@app.callback(
    [Output("chat-response", "children"),
     Output("loading-spinner", "style")],
    [Input("send-button", "n_clicks"),
     Input("user-input", "n_submit")],  #  Pressing Enter works
    [State("user-input", "value"),
     State("supplier-dropdown", "value"),
     State("chat-response", "children")]  # Preserve chat history
)
def ask_sRomancoAI(n_clicks, n_submit, user_query, selected_supplier, chat_history):
    if not user_query:
        return chat_history, {"display": "none"}  # No response if empty query

    # Show spinner while processing
    loading_style = {"display": "block"}

    try:
        selected_data = df[df["supplier_name"] == selected_supplier].iloc[0]
        supplier_info = f"Debt to Equity: {selected_data['debt_to_equity']}, Revenue: {selected_data['revenue']}, Market Cap: {selected_data['market_cap']}, Altman Z-Score: {selected_data['altman_z_score']}, Bankruptcy Risk: {selected_data['bankruptcy_risk']}"

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are SRomancoAI Assistant, an AI expert in financial risk assessment. The user is analyzing supplier data: {supplier_info}"},
                {"role": "user", "content": user_query}
            ]
        )
        ai_reply = response.choices[0].message.content
    except Exception as e:
        ai_reply = f"⚠️ Error: {str(e)}"

    chat_history.append(html.Div(f"🧑‍💻 You: {user_query}", style={"font-weight": "bold", "color": "#00ccff"}))
    chat_history.append(html.Div(f"🤖 SRomancoAI Assistant: {ai_reply}", style={"color": "#ffcc00"}))

    return chat_history, {"display": "none"}  # Hide spinner after response

if __name__ == "__main__":
    app.run_server(debug=True)


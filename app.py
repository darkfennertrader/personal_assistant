import os
import importlib
import streamlit as st
import streamlit_antd_components as sac


HOME_PAGE = "home"
GENERAL_SEARCH = "IBO Chatbot"


with st.sidebar:
    selection = sac.menu(
        [
            sac.MenuItem(
                HOME_PAGE,
                icon="house-fill",
                # tag=[sac.Tag("Tag1", color="green"), sac.Tag("Tag2", "red")],
            ),
            sac.MenuItem(
                "Chatbot Assistants",
                icon="box-fill",
                children=[
                    sac.MenuItem(GENERAL_SEARCH, icon="globe"),
                    # sac.MenuItem(
                    #     "other",
                    #     icon="git",
                    #     description="other items",
                    #     children=[
                    #         sac.MenuItem(
                    #             "google", icon="google", description="item description"
                    #         ),
                    #         sac.MenuItem("gitlab", icon="gitlab"),
                    #         sac.MenuItem("wechat", icon="wechat"),
                    #     ],
                    # ),
                ],
            ),
            # sac.MenuItem("disabled", disabled=True),
            # sac.MenuItem(type="divider"),
            # sac.MenuItem(
            #     "link",
            #     type="group",
            #     children=[
            #         sac.MenuItem(
            #             "my profile",
            #             icon="linkedin",
            #             href="https://www.linkedin.com/in/raimondo-marino/",
            #         ),
            #         sac.MenuItem(
            #             "my repositories",
            #             icon="github",
            #             href="https://github.com/darkfennertrader",
            #         ),
            #     ],
            # ),
        ],
        open_all=True,
        color="pink",
        variant="filled",
    )


# List your pages here:
PAGES = {
    HOME_PAGE: "0_home",
    GENERAL_SEARCH: "1_voice_assistant",
}

# page = sac.pagination(show_total=True, align="center", jump=False, total=len(PAGES))
# st.write(len(PAGES))


# Define function to load a page script
def load_page(page_module_name):
    page_module = importlib.import_module(f"app_pages.{page_module_name}")
    page_module.show()


# Page routing based on the user selection
page_module_name = PAGES[str(selection)]
load_page(page_module_name)

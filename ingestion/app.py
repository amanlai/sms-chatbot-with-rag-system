from pymongo.errors import ServerSelectionTimeoutError
from pymongo.database import Database
import streamlit as st

# from src.password_management import get_hashed_password, check_password
from src import init_connection, add_to_collection


# user_name = getenv('LOGIN_USERNAME')
# hashed_password = get_hashed_password(getenv("LOGIN_PASSWORD"))


def main():

    st.title("Upload Training Data")

    st.markdown("Note: Once a file is uploaded, make sure to click the "
                "accompanying button to upload to the database.")

    reset = st.button("Restart")
    if reset:
        st.session_state["business_name"] = ""
        st.rerun()

    if st.session_state.get("client", None) is None:
        st.session_state["client"] = init_connection()

    # business name
    st.text_input("Enter business name:", key="business_name")
    business_name: str | None = st.session_state["business_name"]
    if business_name:
        # we can restrict the IP addresses that can access the db
        # from the Network Access menu on the MongoDB Atlas Project dashboard
        try:
            db_name = business_name.replace(" ", "")
            db: Database = st.session_state["client"][db_name]
            exists = business_name in db.list_collection_names()
            collection = db[business_name]
        except ServerSelectionTimeoutError:
            st.write("Sorry. Your IP address is not authorized to access "
                     "this app.")
        # file uploader widget
        uploaded_file = st.file_uploader(
            label="Upload a file:",
            type=["pdf", "docx", "doc"]
        )
        add_file = st.button("Add new file to database")
        if uploaded_file and add_file:
            with st.spinner('Reading, splitting and embedding file...'):
                add_to_collection(
                    collection=collection,
                    key=exists,
                    value=uploaded_file
                )
                st.success(
                    "File uploaded, chunked, embedded and indexed "
                    "successfully.\nPlease wait 4-5 mins before using "
                    "the uploaded data as source of truth in the chatbot."
                )


if __name__ == '__main__':
    main()

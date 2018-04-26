import React from "react";
import { navigateTo } from "gatsby-link";
import styled from "styled-components";

const H2 = styled.h2`
  margin: 10px 0 1em 0;
`;

const Form = styled.form`
  display: grid;
  grid-row-gap: 0.5em;
  grid-template-columns: 1fr;
  @media (min-width: 450px) {
    grid-template-columns: 30% 70%;
  }
  width: 100%;
  > label {
    font-style: italic;
    color: grey;
  }
  > input {
    border-radius: 5px;
    -webkit-box-shadow: inset 0px 1px 2px 0px rgba(189,189,189,0.3);
    -moz-box-shadow: inset 0px 1px 2px 0px rgba(189,189,189,0.3);
    box-shadow: inset 1px 1px 2px 0px rgba(189,189,189,0.3);
    border: 1px solid rgba(189,189,189,1);
    padding: 7px;
    :focus {
      outline: none;
      background-color: #eafaf0;
    }
  }
  > textarea {
    height: 150px;
    border-radius: 5px;
    -webkit-box-shadow: inset 0px 1px 2px 0px rgba(189,189,189,0.3);
    -moz-box-shadow: inset 0px 1px 2px 0px rgba(189,189,189,0.3);
    box-shadow: inset 1px 1px 2px 0px rgba(189,189,189,0.3);
    border: 1px solid rgba(189,189,189,1);
    padding: 7px;
    resize: none;
    :focus {
      outline: none;
      background-color: #eafaf0;
    }
  }
  > button {
    width: 120px;
    justify-self: end;
    color: white;
    border-radius: 5px;
    border: 1px solid #1d7f3f;
    background-color: #28aa55;
    cursor: pointer;
    transition: all 0.2s ease;
    padding: 7px
    :hover {
      background-color: #1d7f3f;
    }
  }
`;

function encode(data) {
  return Object.keys(data)
    .map(key => encodeURIComponent(key) + "=" + encodeURIComponent(data[key]))
    .join("&");
}

export default class ContactForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  handleChange = e => {
    this.setState({ [e.target.name]: e.target.value });
  };

  handleSubmit = e => {
    fetch("/", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: encode({ "form-name": "contact", ...this.state })
    })
      .then(() => navigateTo("/thanks/"))
      .catch(error => alert(error));

    e.preventDefault();
  };

  render() {
    return (
      <div>
        <H2>Drop me a line</H2>
        <Form
          name="contact"
          method="post"
          action="/thanks/"
          data-netlify="true"
          data-netlify-honeypot="bot-field"
          onSubmit={this.handleSubmit}
        >
          <label hidden>Don’t fill this out: </label>
          <input hidden name="bot-field" onChange={this.handleChange} />
          <label>Your name:</label>
          <input type="text" name="name" onChange={this.handleChange} />
          <label>Your email:</label>
          <input type="email" name="email" onChange={this.handleChange} />
          <label>Message:</label>
          <textarea name="message" onChange={this.handleChange} />
          <div></div>
          <button type="submit">SEND</button>
        </Form>
      </div>
    );
  }
}

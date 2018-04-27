import React from "react";
import Helmet from "react-helmet";
import styled from "styled-components";
import Img from "gatsby-image";
import FA from "react-fontawesome";

import ContactForm from "../components/ContactForm";

import github from "../images/github-2.svg";
import linkedin from "../images/linkedin-2.svg";
import instagram from "../images/instagram-2.svg";

import resume from "./JAZ-Resume-April-2018.pdf";

const Grid = styled.div`
  display: grid;
  grid-gap: 1em;
  grid-template:
    "avatar title"
    "description description"
    "contact contact"
    "resume resume"
    "social social"
    "form form" / 1fr 2fr;
  @media (min-width: 600px) {
    grid-template:
      "avatar title"
      "contact description"
      "social description"
      "resume form" / 1fr 2fr;
  }
`;

const Avatar = styled(Img)`
  grid-area: avatar;
  margin: auto;
  width: 150px;
  border-radius: 50%;
  border: 1px solid grey;
  @media (max-width: 600px) {
    width: 110px;
  }
`;

const Title = styled.h1`
  grid-area: title;
  align-self: center;
  color: black;
  margin: 0;

  > span {
    color: #28aa55;
  }
`;

const Description = styled.p`
  grid-area: description;
  margin: 0;
  padding-bottom: 1em;
  border-bottom: 1px solid grey;
`;

const Contact = styled.div`
  grid-area: contact;
  margin: auto;
  > h2 {
    margin: 0 0 0.5em 0;
  }
  > p {
    margin: 0 0 0.5em 0;
    color: grey;
  }
`;

const Social = styled.div`
  display: flex;
  justify-content: center;
  grid-area: social;
  border-bottom: 1px solid grey;
  img {
    height: 40px;
    margin: 0 15px 15px 15px;
    transition: all 0.2s ease;
    &:hover {
      opacity: 0.8;
    }
  }
`;

const Form = styled.div`
  grid-area: form;
`;

const Resume = styled.div`
  grid-area: resume;
  justify-self: center;
  margin: 10px 0 20px 0;
  > a {
    color: white;
    border-radius: 5px;
    border: 1px solid #1d7f3f;
    background-color: #28aa55;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.2s ease;
    padding: 7px
    :hover {
      background-color: #1d7f3f;
    }
  }
`;

export default ({ data }) => (
  <div>
    <Helmet>
      <title>About</title>
      <link
        href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"
        rel="stylesheet"
        integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN"
        crossorigin="anonymous"
      />
    </Helmet>
    <Grid>
      <Avatar sizes={data.avatar.sizes} />
      <Title>
        Hi. My name is <span>Josh Zastrow</span>.
      </Title>
      <Description>
        Need an enthusiastic engineer on your data science, machine learning or
        A.I team? Please reach out to me! I am always excited at the prospect of
        collaborating with other passionately driven people on bigger projects.
        Being a nomadic Engineer, I spend a good amount of time abroad, but am
        open for grabbing a coffee if you are in the San Francisco area.
      </Description>
      <Contact>
        <p>
          <FA name="phone-square" /> +1 (240) 418-4040
        </p>
        <p>
          <FA name="envelope-square" /> j.a.zastrow.jr@gmail.com
        </p>
      </Contact>
      <Social>
        <a href="https://github.com/JoshZastrow" target="_blank">
          <img src={github} />
        </a>
        <a
          href="https://www.linkedin.com/in/joshua-zastrow-b8131540/"
          target="_blank"
        >
          <img src={linkedin} />
        </a>
        <a href="https://www.instagram.com/josh.zastrow/?hl=en" target="_blank">
          <img src={instagram} />
        </a>
      </Social>
      <Resume>
        <a href={resume} download="JAZ-Resume-April-2018.pdf">
          Download resume <FA name="arrow-circle-down" />
        </a>
      </Resume>
      <Form>
        <ContactForm />
      </Form>
    </Grid>
  </div>
);

export const pageQuery = graphql`
  query AboutQuery {
    avatar: imageSharp(id: { regex: "/avatar.jpg/" }) {
      sizes(maxWidth: 500, grayscale: false) {
        ...GatsbyImageSharpSizes_tracedSVG
      }
    }
  }
`;
